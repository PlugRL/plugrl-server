from loguru import logger
import torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset
from ..policy.base_policy import InternalState

class RolloutBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer_size, example_internal_state: InternalState):
        self.buffer_size = buffer_size
        self.obs = torch.concatenate([example_internal_state.obs] * buffer_size, dim=0)
        self.actions = torch.concatenate([example_internal_state.action] * buffer_size, dim=0)
        self.logprobs = torch.concatenate([example_internal_state.logprob] * buffer_size, dim=0)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.advantages = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
        self.returns = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)

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
        self.last_rollout_idx = 0
        self.episode_info_buffer = []
        
    def add_infer(self, internal_state: InternalState):
        if self.idx >= self.buffer_size:
            raise IndexError("RolloutBuffer is full")
        
        self.obs[self.idx] = internal_state.obs
        self.actions[self.idx] = internal_state.action
        self.logprobs[self.idx] = internal_state.logprob
        self.values[self.idx] = internal_state.value
        
    def add_feedback(self, reward: float):
        if self.idx >= self.buffer_size:
            raise IndexError("RolloutBuffer is full")
        
        self.rewards[self.idx] = reward
        
        self.idx += 1
        
    def finish_rollout(self, last_value: torch.Tensor, last_done: bool, gamma: float, gae_lambda: float, info: dict):
        last_gae_lam = 0
        for t in reversed(range(self.last_rollout_idx, self.idx)):
            if t == self.idx - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns[self.last_rollout_idx:self.idx] = self.advantages[self.last_rollout_idx:self.idx] + self.values[self.last_rollout_idx:self.idx]
        self.last_rollout_idx = self.idx

        self.episode_info_buffer.append(info)

    def full(self) -> bool:
        return self.idx >= self.buffer_size
    
    def reset(self):
        self.idx = 0
        self.last_rollout_idx = 0
        self.episode_info_buffer = []

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
        
        success, total_reward = 0, 0.0
        for info in self.episode_info_buffer:
            if "is_success" in info:
                success += int(info["is_success"])
            if "episode" in info and "r" in info["episode"]:
                total_reward += float(info["episode"]["r"])
        
        return {
            "losses/explained_variance": explained_var,
            "charts/success_rate": success / len(self.episode_info_buffer) if len(self.episode_info_buffer) > 0 else 0.0,
            "charts/average_reward": total_reward / len(self.episode_info_buffer) if len(self.episode_info_buffer) > 0 else 0.0,
        }