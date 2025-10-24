import uuid
import numpy as np
import torch
from loguru import logger

from vlarl_launcher.buffer.rollout_buffer import RolloutBuffer
from vlarl_launcher.policy.base_policy import InternalState

class GRPODiffusionBuffer(RolloutBuffer): 
    def __init__(
        self, 
        buffer_size: int, 
        example_internal_state: InternalState, 
        gamma: float = 0.99,
        group_size: int = 8,
        epsilon: float = 1e-8,
    ):
        super().__init__(buffer_size, example_internal_state)
        self.gamma = gamma
        self.group_size = group_size
        self.epsilon = epsilon
        self.rets = self.rewards.clone()
    
    def add_frame(
        self, 
        *, 
        prev_node: tuple[int, uuid.UUID], 
        internal_state: InternalState, 
        reward: float, 
        done: bool, 
        last_value: torch.Tensor | None, 
        next_done: bool
    ) -> tuple[int, uuid.UUID]:
        if self.idx >= self.buffer_size:
            return (-1, self.buffer_signature)        
        prev_idx, prev_signature = prev_node
        if prev_signature != self.buffer_signature:
            prev_idx = -1  # Ignore previous index if signature doesn't match         
        current_idx = self.idx
        if prev_idx != -1:
            self.next_indices[prev_idx] = current_idx
            self.rets[current_idx] = self.rets[prev_idx] * self.gamma + reward
        else:
            self.rets[current_idx] = reward
            
        self.obs[current_idx] = internal_state.obs
        self.actions[current_idx] = internal_state.action
        self.logprobs[current_idx] = internal_state.logprob
        self.values[current_idx] = torch.zeros_like(internal_state.value)  # Not used in GRPO
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
        self.next_done[current_idx] = next_done
        
        self.idx += 1
        return (current_idx, self.buffer_signature)
    
    def compute_advantages_and_returns(self):
        episode_starts = self._extract_episode_boundaries()
        self._compute_mc_returns(episode_starts)
        self._compute_group_advantages(episode_starts)
    
    def _extract_episode_boundaries(self) -> list[tuple[int, int]]:
        episode_starts = []
        current_start = 0
        for i in range(self.idx):
            if self.next_done[i]:
                episode_starts.append((current_start, i))
                if i + 1 < self.idx:
                    current_start = i + 1
            elif i + 1 < self.idx and self.next_indices[i] != i + 1:
                episode_starts.append((current_start, i))
                current_start = i + 1
        
        return episode_starts
    
    def _compute_mc_returns(self, episode_starts: list[tuple[int, int]]):
        """
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(T-t)*r_T
        """
        self.returns.zero_()
        for start_idx, end_idx in episode_starts:
            episode_return = 0.0
            for step in reversed(range(start_idx, end_idx + 1)):
                episode_return = self.rewards[step] + self.gamma * episode_return
                self.returns[step] = episode_return
        if episode_starts and episode_starts[-1][1] < self.idx - 1:
            incomplete_start = episode_starts[-1][1] + 1
            episode_return = 0.0
            for step in reversed(range(incomplete_start, self.idx)):
                episode_return = self.rewards[step] + self.gamma * episode_return
                self.returns[step] = episode_return
        elif not episode_starts and self.idx > 0:
            episode_return = 0.0
            for step in reversed(range(0, self.idx)):
                episode_return = self.rewards[step] + self.gamma * episode_return
                self.returns[step] = episode_return
    
    def _compute_group_advantages(self, episode_starts: list[tuple[int, int]]):
        """
        A_i = (G_i - mean(G_group)) / std(G_group)
        """
        if len(episode_starts) == 0:
            logger.warning("No complete episodes to compute advantages")
            self.advantages[:self.idx] = 0.0
            return
        
        num_episodes = len(episode_starts)
        episode_returns = []
        episode_ranges = []
        for start_idx, end_idx in episode_starts:
            episode_returns.append(self.returns[start_idx].item())
            episode_ranges.append((start_idx, end_idx))
        episode_returns = np.array(episode_returns)
        
        # Split episodes into groups
        num_groups = (num_episodes + self.group_size - 1) // self.group_size  # Ceiling division
        for group_idx in range(num_groups):
            group_start = group_idx * self.group_size
            group_end = min((group_idx + 1) * self.group_size, num_episodes)
            # Compute group statistics
            group_returns = episode_returns[group_start:group_end]
            group_mean = group_returns.mean()
            group_std = group_returns.std() + self.epsilon
            # Assign advantages to all steps in the group's episodes
            for ep_idx in range(group_start, group_end):
                start_idx, end_idx = episode_ranges[ep_idx]
                advantage = (episode_returns[ep_idx] - group_mean) / group_std
                self.advantages[start_idx:end_idx + 1] = advantage
    
    def reset(self):
        super().reset()
        self.rets.zero_()
    
    def description(self):
        if self.idx == 0:
            return {
                "buffer/num_episodes": 0,
                "buffer/mean_episode_length": 0.0,
                "buffer/mean_return": 0.0,
            }
        
        episode_starts = self._extract_episode_boundaries()
        episode_lengths = [end - start + 1 for start, end in episode_starts]
        episode_returns = [self.returns[start].item() for start, end in episode_starts]
        
        return {
            "buffer/num_episodes": len(episode_starts),
            "buffer/mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "buffer/mean_return": np.mean(episode_returns) if episode_returns else 0.0,
            "buffer/std_return": np.std(episode_returns) if episode_returns else 0.0,
            "buffer/min_return": np.min(episode_returns) if episode_returns else 0.0,
            "buffer/max_return": np.max(episode_returns) if episode_returns else 0.0,
            "buffer/utilization": self.idx / self.buffer_size,
        }