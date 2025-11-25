from loguru import logger
import torch
import torch.utils.data
import tensordict
import uuid

from vlarl_launcher.policy.base_policy import InternalState
from vlarl_launcher.common.tensor_container import TensorContainer, tensor_container

@tensor_container
class ReplayBufferSamples(TensorContainer):
    obs: torch.Tensor | tensordict.TensorDict
    next_obs: torch.Tensor | tensordict.TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor

class ReplayBuffer(torch.utils.data.Dataset):
    obs: torch.Tensor | tensordict.TensorDict
    next_obs: torch.Tensor | tensordict.TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor
    
    def __init__(self, buffer_size, example_internal_state: InternalState):
        self.buffer_size = buffer_size
        self.obs = tensordict.stack([example_internal_state.obs[0]] * buffer_size, dim=0)
        self.next_obs = tensordict.stack([example_internal_state.obs[0]] * buffer_size, dim=0)
        self.actions = torch.concatenate([example_internal_state.action] * buffer_size, dim=0)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.timeouts = torch.zeros(buffer_size, dtype=torch.bool)
        
        logger.info(f"""
            Initialized ReplayBuffer with buffer_size={buffer_size}
            obs shape: {self.obs.shape}
            next_obs shape: {self.next_obs.shape}
            actions shape: {self.actions.shape}
            rewards shape: {self.rewards.shape}
            dones shape: {self.dones.shape}
            timeouts shape: {self.timeouts.shape}
        """)
        
        self.idx = 0
        self._full = False
        self.buffer_signature = uuid.uuid4()

    def add_frame(self, 
        *, 
        prev_node: tuple[int, uuid.UUID], 
        obs: torch.Tensor | tensordict.TensorDict,
        next_obs: torch.Tensor | tensordict.TensorDict,
        action: torch.Tensor,
        reward: float, 
        done: bool,
        timeout: bool,
    ) -> tuple[int, uuid.UUID]: 
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
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
            obs=self.obs[batch_inds],
            next_obs=self.next_obs[batch_inds],
            actions=self.actions[batch_inds],
            rewards=self.rewards[batch_inds],
            dones=self.dones[batch_inds],
            timeouts=self.timeouts[batch_inds],
        )
        
    def __len__(self) -> int:
        return self.buffer_size if self._full else self.idx
    
    def full(self) -> bool:
        return self._full