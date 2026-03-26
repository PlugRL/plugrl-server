import uuid
from typing import Any

import torch
import numpy as np
from loguru import logger

from plugrl_server.common.data_utils import (
    numpy_tree_to_torch,
    stack_numpy_tree,
)
from plugrl_server.buffer.numpy_tree_storage import NumpyTreeStorage
from plugrl_server.policy.state import PolicyTrainState


def _require_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value)!r}.")
    return value


class RolloutBuffer(torch.utils.data.Dataset):
    obs_storage: NumpyTreeStorage
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    last_values: np.ndarray
    next_done: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    dones: np.ndarray
    next_indices: np.ndarray
    idx: int
    buffer_signature: uuid.UUID
    buffer_size: int
    episode_info_buffer: list[dict[str, Any]]

    def __init__(
        self,
        buffer_size,
        example_train_state: PolicyTrainState,
    ):
        if example_train_state is None:
            raise ValueError("RolloutBuffer requires non-empty example_train_state.")

        self.buffer_size = buffer_size

        self.obs_storage = NumpyTreeStorage.from_example(
            example_train_state["obs"], buffer_size
        )

        example_action = _require_array(example_train_state["action"], "train_state['action']")
        example_value = _require_array(example_train_state["value"], "train_state['value']")
        example_logprob = _require_array(example_train_state["logprob"], "train_state['logprob']")

        action_shape = example_action.shape[1:]
        value_shape = example_value.shape[1:]
        logprob_shape = example_logprob.shape[1:]

        self.actions = np.empty(
            (buffer_size,) + action_shape, dtype=example_action.dtype
        )
        self.logprobs = np.empty(
            (buffer_size,) + logprob_shape, dtype=example_logprob.dtype
        )

        self.values = np.empty(
            (buffer_size,) + value_shape, dtype=example_value.dtype
        )
        self.last_values = np.empty(
            (buffer_size,) + value_shape, dtype=example_value.dtype
        )
        self.advantages = np.empty(
            (buffer_size,) + value_shape, dtype=example_value.dtype
        )
        self.returns = np.empty(
            (buffer_size,) + value_shape, dtype=example_value.dtype
        )

        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_done = np.zeros(buffer_size, dtype=np.bool_)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.next_indices = np.zeros(buffer_size, dtype=np.int32)

        logger.info(f"""
            Initialized RolloutBuffer with buffer_size={buffer_size}
            obs shape: {_describe_tree_shape(self.obs_storage.data)}
            actions shape: {self.actions.shape}
            logprobs shape: {self.logprobs.shape}
            rewards shape: {self.rewards.shape}
            values shape: {self.values.shape}
            advantages shape: {self.advantages.shape}
        """)

        self.idx = 0
        self.buffer_signature = uuid.uuid4()
        self.episode_info_buffer = []

    def add_frame(
        self,
        *,
        prev_node: tuple[int, uuid.UUID],
        train_state: PolicyTrainState,
        reward: float,
        done: bool,
        last_value: np.ndarray | None,
        next_done: bool,
    ) -> tuple[int, uuid.UUID]:
        if train_state is None:
            raise ValueError("train_state must not be None")
        if self.idx >= self.buffer_size:
            return (-1, self.buffer_signature)
        prev_idx, prev_signature = prev_node
        if prev_signature != self.buffer_signature:
            prev_idx = -1  # Ignore previous index if signature doesn't match
        current_idx = self.idx
        if prev_idx != -1:
            self.next_indices[prev_idx] = current_idx
        self.obs_storage.set_item(current_idx, train_state["obs"])
        self.actions[current_idx] = _require_array(train_state["action"], "train_state['action']")[0]
        self.logprobs[current_idx] = _require_array(train_state["logprob"], "train_state['logprob']")[0]
        self.values[current_idx] = _require_array(train_state["value"], "train_state['value']")[0]
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
        if last_value is not None:
            self.last_values[current_idx] = last_value
        self.next_done[current_idx] = next_done

        self.idx += 1
        return (current_idx, self.buffer_signature)

    def finish_rollout(self, info: dict):
        self.episode_info_buffer.append(info)

    def full(self) -> bool:
        return self.idx >= self.buffer_size

    def reset(self):
        self.idx = 0
        self.buffer_signature = uuid.uuid4()
        self.episode_info_buffer = []
        self.next_indices.fill(0)

    def __len__(self):
        return self.idx

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= self.idx:
            raise IndexError("RolloutBuffer index out of range")
        return (
            self.obs_storage.get_item(idx),
            self.actions[idx],
            self.logprobs[idx],
            self.rewards[idx],
            self.values[idx],
            self.advantages[idx],
            self.returns[idx],
        )

    def description(self):
        y_pred, y_true = self.values[: self.idx], self.returns[: self.idx]
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "losses/explained_variance": explained_var,
        }

    def collate_fn(self, batch: list[tuple]) -> tuple:
        obs_items, *rest = zip(*batch)
        return (
            numpy_tree_to_torch(stack_numpy_tree(list(obs_items), axis=0)),
            *(torch.from_numpy(np.stack(items, axis=0)) for items in rest),
        )

    def as_dict(self) -> dict:
        idx = self.idx
        obs_slice = self.obs_storage.get_item(slice(None, idx))
        data = dict(
            obs=obs_slice,
            actions=self.actions[:idx].copy(),
            logprobs=self.logprobs[:idx].copy(),
            rewards=self.rewards[:idx].copy(),
            values=self.values[:idx].copy(),
            advantages=self.advantages[:idx].copy(),
            returns=self.returns[:idx].copy(),
            dones=self.dones[:idx].copy(),
            next_indices=self.next_indices[:idx].copy(),
            idx=idx,
            buffer_signature=self.buffer_signature,
            episode_info_buffer=self.episode_info_buffer.copy(),
            obs_spec=self.obs_storage.spec,
        )
        return data

    def load_dict(self, data: dict) -> None:
        self.idx = data["idx"]
        self.obs_storage = NumpyTreeStorage(spec=data["obs_spec"], capacity=self.buffer_size)
        self.obs_storage.set_item(slice(None, self.idx), data["obs"])
        self.actions[: self.idx] = data["actions"]
        self.logprobs[: self.idx] = data["logprobs"]
        self.rewards[: self.idx] = data["rewards"]
        self.values[: self.idx] = data["values"]
        self.advantages[: self.idx] = data["advantages"]
        self.returns[: self.idx] = data["returns"]
        self.dones[: self.idx] = data["dones"]

        # [WARNING] next_indices is not a tensor, and for some reason it is read-only so we just ignore it here

def _describe_tree_shape(value) -> str:
    if isinstance(value, np.ndarray):
        return str(tuple(value.shape))
    return str(dict((key, _describe_tree_shape(item)) for key, item in value.items()))


class GAEBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size,
        example_train_state: PolicyTrainState,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__(buffer_size, example_train_state)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages_and_returns(self):
        for step in reversed(range(self.idx)):
            next_idx = self.next_indices[step]
            next_gae_lam = self.advantages[next_idx] if next_idx != 0 else 0
            if next_idx == 0:
                next_non_terminal = 1.0 - float(self.next_done[step])
                next_values = self.last_values[step]
                # check last values not overflow or abs extreme large
                assert abs(next_values).max() < 1e6, (
                    f"last_values overflow: {next_values}"
                )
            else:
                next_non_terminal = 1.0 - float(self.dones[next_idx])
                next_values = self.values[next_idx]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            self.advantages[step] = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * next_gae_lam
            )

        self.returns = self.advantages + self.values
