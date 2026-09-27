import uuid

import numpy as np

from plugrl_server.algorithm.dppo.third_party.reward_scaling import RunningMeanStd
from plugrl_server.buffer.rollout_buffer import GAEBuffer
from plugrl_server.policy.base_policy import BasePolicy
from plugrl_server.policy.state import PolicyTrainState


class PPOBuffer(GAEBuffer):
    """GAE, with rewards scaled the way gymnasium's NormalizeReward scales them.

    Each frame's discounted return is kept as it arrives; at learning time the
    running variance takes in the buffer's returns and every reward is divided
    by the running deviation, then clipped. NormalizeReward divides each reward
    by the deviation as it stood at that step instead; over the same returns
    the two differ only in how recent the estimate is.

    The discounted return restarts with each episode, as it does in
    NormalizeReward and in DPPO's RunningRewardScaler. `DPPOBuffer` follows
    the server's chain of frames, which runs on across episodes, so its return
    carries over from one episode into the next.
    """

    def __init__(
        self,
        buffer_size,
        example_train_state: PolicyTrainState,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_rewards: bool = True,
        reward_clip: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__(buffer_size, example_train_state, gamma, gae_lambda)
        self.normalize_rewards = normalize_rewards
        self.reward_clip = reward_clip
        self.epsilon = epsilon
        self.ret_rms = RunningMeanStd(shape=())
        self.rets = np.zeros(buffer_size, dtype=np.float64)

    def add_frame(
        self,
        *,
        prev_node: tuple[int, uuid.UUID],
        train_state: PolicyTrainState,
        reward: float,
        terminated: bool,
        truncated: bool,
        last_value: np.ndarray | None,
        next_terminated: bool,
        next_truncated: bool,
    ) -> tuple[int, uuid.UUID]:
        node = super().add_frame(
            prev_node=prev_node,
            train_state=train_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            last_value=last_value,
            next_terminated=next_terminated,
            next_truncated=next_truncated,
        )
        current_idx = node[0]
        if current_idx == -1:
            return node
        prev_idx, prev_signature = prev_node
        # `terminated` or `truncated` on a frame means the step before it
        # ended an episode, so this frame begins one.
        continues = (
            prev_idx != -1
            and prev_signature == self.buffer_signature
            and not (terminated or truncated)
        )
        self.rets[current_idx] = float(reward) + (
            self.gamma * self.rets[prev_idx] if continues else 0.0
        )
        return node

    def compute_advantages_and_returns(
        self, policy: BasePolicy | None = None, batch_size: int = 1
    ):
        if self.normalize_rewards and self.idx > 0:
            self.ret_rms.update(self.rets[: self.idx])
            scale = np.float32(np.sqrt(self.ret_rms.var + self.epsilon))
            self.rewards[: self.idx] = np.clip(
                self.rewards[: self.idx] / scale, -self.reward_clip, self.reward_clip
            )
        return super().compute_advantages_and_returns(
            policy=policy, batch_size=batch_size
        )
