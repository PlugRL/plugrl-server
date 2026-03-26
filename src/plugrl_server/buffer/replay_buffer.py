import dataclasses
import uuid

import torch

from plugrl_server.common.data_utils import (
    TorchTree,
    create_empty_torch_tree,
    torch_tree_get_item,
    torch_tree_set_item,
)
from plugrl_server.policy.state_adapter import TrainStateLike, train_state_to_tensors


@dataclasses.dataclass(frozen=True)
class ReplayBufferSamples:
    obs: TorchTree
    next_obs: TorchTree
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor


class ReplayBuffer(torch.utils.data.Dataset):
    obs: TorchTree
    next_obs: TorchTree
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor

    def __init__(self, buffer_size, example_train_state: TrainStateLike):
        self.buffer_size = buffer_size
        example_train_tensors = train_state_to_tensors(example_train_state)

        sample_obs = torch_tree_get_item(example_train_tensors.obs, 0)
        self.obs = create_empty_torch_tree(sample_obs, buffer_size)
        self.next_obs = create_empty_torch_tree(sample_obs, buffer_size)

        self.actions = torch.empty(
            (buffer_size,) + example_train_tensors.action.shape[1:],
            dtype=example_train_tensors.action.dtype,
        )
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.timeouts = torch.zeros(buffer_size, dtype=torch.bool)

        self.idx = 0
        self._full = False
        self.buffer_signature = uuid.uuid4()

    def add_frame(
        self,
        *,
        prev_node: tuple[int, uuid.UUID],
        obs: TorchTree,
        next_obs: TorchTree,
        action: torch.Tensor,
        reward: float,
        done: bool,
        timeout: bool,
    ) -> tuple[int, uuid.UUID]:
        torch_tree_set_item(self.obs, self.idx, obs)
        torch_tree_set_item(self.next_obs, self.idx, next_obs)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.timeouts[self.idx] = timeout

        self.idx += 1
        if self.idx == self.buffer_size:
            self._full = True
            self.idx = 0  # Overwrite old data once buffer is full

        return (-1, self.buffer_signature)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if self._full:
            batch_inds = torch.randint(0, self.buffer_size, (batch_size,))
        else:
            batch_inds = torch.randint(0, self.idx, (batch_size,))

        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: torch.Tensor) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            obs=torch_tree_get_item(self.obs, batch_inds),
            next_obs=torch_tree_get_item(self.next_obs, batch_inds),
            actions=self.actions[batch_inds],
            rewards=self.rewards[batch_inds],
            dones=self.dones[batch_inds],
            timeouts=self.timeouts[batch_inds],
        )

    def __len__(self) -> int:
        return self.buffer_size if self._full else self.idx

    def full(self) -> bool:
        return self._full
