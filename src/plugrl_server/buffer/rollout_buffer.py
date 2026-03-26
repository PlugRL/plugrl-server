import uuid
from typing import Any

import torch
import numpy as np
from loguru import logger

from plugrl_server.common.data_utils import (
    TorchTree,
    create_empty_torch_tree,
    numpy_tree_to_torch,
    stack_torch_tree,
    torch_tree_get_item,
    torch_tree_set_item,
    torch_tree_to_numpy,
)
from plugrl_server.policy.state_adapter import TrainStateLike, train_state_to_tensors


class RolloutBuffer(torch.utils.data.Dataset):
    obs: TorchTree
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    last_values: torch.Tensor
    next_done: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    dones: torch.Tensor
    next_indices: np.ndarray
    idx: int
    buffer_signature: uuid.UUID
    buffer_size: int
    episode_info_buffer: list[dict[str, Any]]

    def __init__(
        self,
        buffer_size,
        example_train_state: TrainStateLike,
    ):
        example_train_tensors = train_state_to_tensors(example_train_state)
        sample_obs = torch_tree_get_item(example_train_tensors.obs, 0)

        self.buffer_size = buffer_size

        self.obs = create_empty_torch_tree(sample_obs, buffer_size)

        action_shape = example_train_tensors.action.shape[1:]
        value_shape = example_train_tensors.value.shape[1:]
        logprob_shape = example_train_tensors.logprob.shape[1:]

        self.actions = torch.empty(
            (buffer_size,) + action_shape, dtype=example_train_tensors.action.dtype
        )
        self.logprobs = torch.empty(
            (buffer_size,) + logprob_shape, dtype=example_train_tensors.logprob.dtype
        )

        self.values = torch.empty(
            (buffer_size,) + value_shape, dtype=example_train_tensors.value.dtype
        )
        self.last_values = torch.empty(
            (buffer_size,) + value_shape, dtype=example_train_tensors.value.dtype
        )
        self.advantages = torch.empty(
            (buffer_size,) + value_shape, dtype=example_train_tensors.value.dtype
        )
        self.returns = torch.empty(
            (buffer_size,) + value_shape, dtype=example_train_tensors.value.dtype
        )

        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.next_done = torch.zeros(buffer_size, dtype=torch.bool)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.next_indices = np.zeros(buffer_size, dtype=np.int32)

        logger.info(f"""
            Initialized RolloutBuffer with buffer_size={buffer_size}
            obs shape: {_describe_tree_shape(self.obs)}
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
        train_state: TrainStateLike,
        reward: float,
        done: bool,
        last_value: torch.Tensor | None,
        next_done: bool,
    ) -> tuple[int, uuid.UUID]:
        train_state_tensors = train_state_to_tensors(train_state)
        if self.idx >= self.buffer_size:
            return (-1, self.buffer_signature)
        prev_idx, prev_signature = prev_node
        if prev_signature != self.buffer_signature:
            prev_idx = -1  # Ignore previous index if signature doesn't match
        current_idx = self.idx
        if prev_idx != -1:
            self.next_indices[prev_idx] = current_idx
        torch_tree_set_item(
            self.obs,
            current_idx,
            torch_tree_get_item(train_state_tensors.obs, 0),
        )
        self.actions[current_idx] = train_state_tensors.action
        self.logprobs[current_idx] = train_state_tensors.logprob
        self.values[current_idx] = train_state_tensors.value
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
            torch_tree_get_item(self.obs, idx),
            self.actions[idx],
            self.logprobs[idx],
            self.rewards[idx],
            self.values[idx],
            self.advantages[idx],
            self.returns[idx],
        )

    def description(self):
        y_pred, y_true = self.values.cpu().numpy(), self.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "losses/explained_variance": explained_var,
        }

    def collate_fn(self, batch: list[tuple]) -> tuple:
        obs_items, *rest = zip(*batch)
        return (
            stack_torch_tree(list(obs_items), dim=0),
            *(torch.stack(items, dim=0) for items in rest),
        )

    def as_dict(self) -> dict:
        idx = self.idx
        obs_slice = torch_tree_to_numpy(torch_tree_get_item(self.obs, slice(None, idx)))
        data = dict(
            obs=obs_slice,
            actions=self.actions[:idx].cpu().numpy(),
            logprobs=self.logprobs[:idx].cpu().numpy(),
            rewards=self.rewards[:idx].cpu().numpy(),
            values=self.values[:idx].cpu().numpy(),
            advantages=self.advantages[:idx].cpu().numpy(),
            returns=self.returns[:idx].cpu().numpy(),
            dones=self.dones[:idx].cpu().numpy(),
            next_indices=self.next_indices[:idx].copy(),
            idx=idx,
            buffer_signature=self.buffer_signature,
            episode_info_buffer=self.episode_info_buffer.copy(),
        )
        return data

    def load_dict(self, data: dict) -> None:
        self.idx = data["idx"]
        torch_tree_set_item(
            self.obs,
            slice(None, self.idx),
            numpy_tree_to_torch(data["obs"]),
        )
        self.actions[: self.idx] = torch.from_numpy(data["actions"])
        self.logprobs[: self.idx] = torch.from_numpy(data["logprobs"])
        self.rewards[: self.idx] = torch.from_numpy(data["rewards"])
        self.values[: self.idx] = torch.from_numpy(data["values"])
        self.advantages[: self.idx] = torch.from_numpy(data["advantages"])
        self.returns[: self.idx] = torch.from_numpy(data["returns"])
        self.dones[: self.idx] = torch.from_numpy(data["dones"])

        # [WARNING] next_indices is not a tensor, and for some reason it is read-only so we just ignore it here


def _describe_tree_shape(value: TorchTree) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return str(dict((key, _describe_tree_shape(item)) for key, item in value.items()))


class GAEBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size,
        example_train_state: TrainStateLike,
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
