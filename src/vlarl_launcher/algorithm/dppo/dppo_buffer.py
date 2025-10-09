import uuid
import numpy as np
import torch
from dppo.util.reward_scaling import RunningMeanStd

from vlarl_launcher.buffer.rollout_buffer import GAEBuffer
from vlarl_launcher.policy.base_policy import InternalState

class DPPOBuffer(GAEBuffer):
    def __init__(
        self, buffer_size, example_internal_state: InternalState, 
        gamma: float = 0.99, gae_lambda: float = 0.95,
        cliprew: float = 10., epsilon: float = 1e-8,
    ):
        super().__init__(buffer_size, example_internal_state)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ret_rms = RunningMeanStd(shape=()) # per env = true
        self.cliprew = cliprew
        self.epsilon = epsilon
        self.rets = self.rewards.clone()
    
    def add_frame(self, *, prev_node: tuple[int, uuid.UUID], internal_state: InternalState, reward: float, done: bool, last_value: torch.Tensor | None, next_done: bool) -> tuple[int, uuid.UUID]:        
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
        self.values[current_idx] = internal_state.value
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
        if last_value is not None: self.last_values[current_idx] = last_value
        self.next_done[current_idx] = next_done

        self.idx += 1
        return (current_idx, self.buffer_signature)
    
    def compute_advantages_and_returns(self):
        # Normalize rewards
        rets = self.rets[:self.idx].cpu().numpy()
        self.ret_rms.update(rets)
        self.rewards[:self.idx] = torch.clamp(
            self.rewards[:self.idx] / torch.sqrt(torch.tensor(self.ret_rms.var + self.epsilon).float()), 
            -self.cliprew, self.cliprew
        )
        
        return super().compute_advantages_and_returns()