import uuid
from collections import deque
import numpy as np
import torch
from dppo.util.reward_scaling import RunningMeanStd

from plugrl_server.buffer.rollout_buffer import GAEBuffer
from plugrl_server.policy.base_policy import BasePolicy
from plugrl_server.policy.state import PolicyTrainState
from plugrl_server.common.data_utils import batch_aggregate
from .train_state import as_dppo_train_state


class DPPOBuffer(GAEBuffer):
    def __init__(
        self,
        buffer_size,
        example_train_state: PolicyTrainState,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        cliprew: float = 10.0,
        epsilon: float = 1e-8,
        use_normalized_rewards: bool = True,
    ):
        super().__init__(buffer_size, example_train_state)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ret_rms = RunningMeanStd(shape=())  # per env = true
        self.cliprew = cliprew
        self.epsilon = epsilon
        self.use_normalized_rewards = use_normalized_rewards
        self.rets = np.zeros_like(self.rewards)
        self.next_obs_value_requests = deque()

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
        dppo_train_state = as_dppo_train_state(train_state)
        if self.idx >= self.buffer_size:
            return (-1, self.buffer_signature)
        prev_idx, prev_signature = prev_node
        if prev_signature != self.buffer_signature:
            prev_idx = -1  # Ignore previous index if signature doesn't match
        current_idx = self.idx
        if prev_idx != -1:
            self.next_indices[prev_idx] = current_idx
            self.rets[current_idx] = self.rets[prev_idx] * self.gamma + float(reward)
        else:
            self.rets[current_idx] = float(reward)
        self.obs_storage.set_item(
            current_idx,
            dict(
                x=dppo_train_state.obs.x,
                t=dppo_train_state.obs.t,
                cond=dict(state=dppo_train_state.obs.cond.state),
            ),
        )
        self.actions[current_idx] = dppo_train_state.action[0]
        self.logprobs[current_idx] = dppo_train_state.logprob[0]
        self.values[current_idx] = dppo_train_state.value[0]
        self.rewards[current_idx] = float(reward)
        self.dones[current_idx] = bool(done)
        if last_value is not None:
            self.last_values[current_idx] = last_value
        self.next_done[current_idx] = bool(next_done)
        self.idx += 1
        return (current_idx, self.buffer_signature)

    def add_next_obs_value_request(self, *, obs: dict, end_node: tuple[int, uuid.UUID]):
        if end_node[0] != -1:
            self.next_obs_value_requests.append((obs, end_node))
            assert self.next_indices[end_node[0]] == 0, (
                "Next index for end_node should be unset."
            )
            while self.next_indices[self.next_obs_value_requests[0][1][0]] != 0:
                self.next_obs_value_requests.popleft()

    def compute_advantages_and_returns(
        self, policy: BasePolicy | None = None, batch_size: int = 1
    ):
        # Normalize rewards
        rets = self.rets[: self.idx]
        self.ret_rms.update(rets)
        if self.use_normalized_rewards:
            scale = np.float32(np.sqrt(self.ret_rms.var + self.epsilon))
            self.rewards[: self.idx] = np.clip(
                self.rewards[: self.idx] / scale,
                -self.cliprew,
                self.cliprew,
            )

        next_ids = []
        next_observations = []
        for obs, node in self.next_obs_value_requests:
            idx, signature = node
            if signature == self.buffer_signature and self.next_indices[idx] == 0:
                next_ids.append(int(idx))
                next_observations.append(obs)
        self.next_obs_value_requests.clear()
        if len(next_observations) > 0:
            assert policy is not None, (
                "Policy must be provided to compute values for done observations."
            )
            for i in range(0, len(next_observations), batch_size):
                batch_obs = batch_aggregate(
                    next_observations[i : i + batch_size], aggregate_method="concat"
                )
                with torch.inference_mode():
                    batch_values = policy.get_value(batch_obs).cpu().numpy()
                self.last_values[next_ids[i : i + batch_size]] = batch_values

        return super().compute_advantages_and_returns()
