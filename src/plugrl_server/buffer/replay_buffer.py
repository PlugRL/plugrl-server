import dataclasses
import uuid

import numpy as np
import torch

from plugrl_server.buffer.numpy_tree_storage import NumpyTreeStorage
from plugrl_server.common.data_utils import (
    NumpyTree,
    TorchTree,
    numpy_tree_to_torch,
)

REPLAY_BUFFER_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ReplayBufferSamples:
    obs: TorchTree
    next_obs: TorchTree
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor

    def to(self, device: torch.device) -> "ReplayBufferSamples":
        return ReplayBufferSamples(
            obs=_torch_tree_to_device(self.obs, device),
            next_obs=_torch_tree_to_device(self.next_obs, device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            timeouts=self.timeouts.to(device),
        )


class ReplayBuffer(torch.utils.data.Dataset):
    buffer_size: int
    obs_storage: NumpyTreeStorage
    next_obs_storage: NumpyTreeStorage
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    idx: int
    _full: bool
    buffer_signature: uuid.UUID

    def __init__(self, buffer_size: int, example_obs: NumpyTree, example_action: np.ndarray):
        super().__init__()
        self.buffer_size = buffer_size
        self.obs_storage = NumpyTreeStorage.from_example(example_obs, buffer_size)
        self.next_obs_storage = NumpyTreeStorage.from_example(example_obs, buffer_size)

        self.actions = np.empty(
            (buffer_size,) + tuple(example_action.shape),
            dtype=example_action.dtype,
        )
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.timeouts = np.zeros(buffer_size, dtype=np.bool_)

        self.idx = 0
        self._full = False
        self.buffer_signature = uuid.uuid4()

    def add_frame(
        self,
        *,
        prev_node: tuple[int, uuid.UUID],
        obs: NumpyTree,
        next_obs: NumpyTree,
        action: np.ndarray,
        reward: float,
        done: bool,
        timeout: bool,
    ) -> tuple[int, uuid.UUID]:
        self.obs_storage.set_item(self.idx, obs)
        self.next_obs_storage.set_item(self.idx, next_obs)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.timeouts[self.idx] = timeout

        self.idx += 1
        if self.idx == self.buffer_size:
            self._full = True
            self.idx = 0

        return (-1, self.buffer_signature)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper = self.buffer_size if self._full else self.idx
        batch_inds = np.random.randint(0, upper, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            obs=numpy_tree_to_torch(self.obs_storage.get_item(batch_inds)),
            next_obs=numpy_tree_to_torch(self.next_obs_storage.get_item(batch_inds)),
            actions=torch.from_numpy(self.actions[batch_inds]),
            rewards=torch.from_numpy(self.rewards[batch_inds]),
            dones=torch.from_numpy(self.dones[batch_inds]),
            timeouts=torch.from_numpy(self.timeouts[batch_inds]),
        )

    def __len__(self) -> int:
        return self.buffer_size if self._full else self.idx

    def full(self) -> bool:
        return self._full

    def as_dict(self) -> dict:
        idx = self.buffer_size if self._full else self.idx
        return dict(
            buffer_schema_version=REPLAY_BUFFER_SCHEMA_VERSION,
            obs=self.obs_storage.get_item(slice(None, idx)),
            next_obs=self.next_obs_storage.get_item(slice(None, idx)),
            actions=self.actions[:idx].copy(),
            rewards=self.rewards[:idx].copy(),
            dones=self.dones[:idx].copy(),
            timeouts=self.timeouts[:idx].copy(),
            idx=self.idx,
            full=self._full,
            buffer_signature=self.buffer_signature,
            obs_spec=self.obs_storage.spec,
        )

    def load_dict(self, data: dict) -> None:
        schema_version = data.get("buffer_schema_version", 0)
        if schema_version != REPLAY_BUFFER_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported replay buffer schema version: "
                f"{schema_version}, expected {REPLAY_BUFFER_SCHEMA_VERSION}."
            )
        self.idx = data["idx"]
        self._full = data["full"]
        self.buffer_signature = data["buffer_signature"]
        self.obs_storage = NumpyTreeStorage(spec=data["obs_spec"], capacity=self.buffer_size)
        self.next_obs_storage = NumpyTreeStorage(
            spec=data["obs_spec"], capacity=self.buffer_size
        )
        upper = self.buffer_size if self._full else self.idx
        self.obs_storage.set_item(slice(None, upper), data["obs"])
        self.next_obs_storage.set_item(slice(None, upper), data["next_obs"])
        self.actions[:upper] = data["actions"]
        self.rewards[:upper] = data["rewards"]
        self.dones[:upper] = data["dones"]
        self.timeouts[:upper] = data["timeouts"]


def _torch_tree_to_device(tree: TorchTree, device: torch.device) -> TorchTree:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    return dict((key, _torch_tree_to_device(value, device)) for key, value in tree.items())
