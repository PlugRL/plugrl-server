from loguru import logger
import torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset
from ..policy.base_policy import InternalState
import uuid

class RolloutBuffer(torch.utils.data.Dataset):
    obs: torch.Tensor
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
    
    def __init__(self, buffer_size, example_internal_state: InternalState):
        self.buffer_size = buffer_size
        self.obs = torch.concatenate([example_internal_state.obs] * buffer_size, dim=0)
        self.actions = torch.concatenate([example_internal_state.action] * buffer_size, dim=0)
        self.logprobs = torch.concatenate([example_internal_state.logprob] * buffer_size, dim=0)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.last_values = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.next_done = torch.zeros(buffer_size, dtype=torch.bool)
        self.advantages = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.returns = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.next_indices = np.zeros(buffer_size, dtype=np.int32)

        logger.info(f"""
            Initialized RolloutBuffer with buffer_size={buffer_size}
            obs shape: {self.obs.shape}
            actions shape: {self.actions.shape}
            logprobs shape: {self.logprobs.shape}
            rewards shape: {self.rewards.shape}
            values shape: {self.values.shape}
            advantages shape: {self.advantages.shape}
        """)
        
        self.idx = 0
        self.buffer_signature = uuid.uuid4()
        self.episode_info_buffer = []
    
    def add_frame(self, *, prev_node: tuple[int, uuid.UUID], internal_state: InternalState, reward: float, done: bool, last_value: torch.Tensor | None, next_done: bool) -> tuple[int, uuid.UUID]:        
        if self.idx >= self.buffer_size:
            return (-1, self.buffer_signature) 
        
        prev_idx, prev_signature = prev_node
        if prev_signature != self.buffer_signature:
            prev_idx = -1  # Ignore previous index if signature doesn't match
        current_idx = self.idx
        if prev_idx != -1:
            self.next_indices[prev_idx] = current_idx
        self.obs[current_idx] = internal_state.obs
        self.actions[current_idx] = internal_state.action
        self.logprobs[current_idx] = internal_state.logprob
        self.values[current_idx] = internal_state.value
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
        if last_value is not None: self.last_values[current_idx] = last_value
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
            self.obs[idx],
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
        
class GAEBuffer(RolloutBuffer):
    def __init__(self, buffer_size, example_internal_state: InternalState, gamma: float = 0.99, gae_lambda: float = 0.95):
        super().__init__(buffer_size, example_internal_state)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_advantages_and_returns(self):
        for step in reversed(range(self.idx)):
            next_idx = self.next_indices[step]
            next_gae_lam = self.advantages[next_idx] if next_idx != 0 else 0
            if next_idx == 0:
                next_non_terminal = 1.0 - float(self.next_done[step])
                next_values = self.last_values[step]
            else:
                next_non_terminal = 1.0 - float(self.dones[next_idx])
                next_values = self.values[next_idx]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * next_gae_lam
            
        self.returns = self.advantages + self.values