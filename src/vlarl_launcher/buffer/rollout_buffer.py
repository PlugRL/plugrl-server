from loguru import logger
import torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset
import tensordict
from ..policy.base_policy import InternalState
import uuid
import os
import psutil
from loguru import logger

def log_cpu_memory_usage(step, phase="unknown"):
    """Log detailed CPU memory usage information for the current process."""
    
    # 获取当前 Python 进程的 ID
    pid = os.getpid()
    
    # 尝试获取进程对象
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.warning("Process not found for memory logging.")
        return

    # 获取进程内存信息 (单位：字节)
    memory_info = process.memory_info()
    
    # 实际使用的物理内存 (Resident Set Size, RSS)
    memory_rss_gb = memory_info.rss / (1024 ** 3)  
    # 进程分配的虚拟内存 (Virtual Memory Size, VMS)
    memory_vms_gb = memory_info.vms / (1024 ** 3)
    
    # 获取系统总内存信息 (可选，提供上下文)
    system_memory = psutil.virtual_memory()
    system_total_gb = system_memory.total / (1024 ** 3)
    system_available_gb = system_memory.available / (1024 ** 3)

    logger.info(
        f"Step {step} ({phase}): CPU memory (Process) - RSS: {memory_rss_gb:.2f}GB, VMS: {memory_vms_gb:.2f}GB | System - Total: {system_total_gb:.2f}GB, Available: {system_available_gb:.2f}GB"
    )
    
def _recursively_create_empty_td(template_td: torch.Tensor, buffer_size):
    new_data = {}
    for key, item in template_td.items():
        if isinstance(item, torch.Tensor): 
            new_shape = (buffer_size,) + item.shape
            new_data[key] = torch.empty(new_shape, dtype=item.dtype, device=item.device)
        elif isinstance(item, tensordict.TensorDict):
            new_data[key] = _recursively_create_empty_td(item, buffer_size)
        else:
             new_data[key] = item
             
    return tensordict.TensorDict(new_data, batch_size=[buffer_size] + list(template_td.shape))

class RolloutBuffer(torch.utils.data.Dataset):
    obs: torch.Tensor | tensordict.TensorDict
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
        sample_obs = example_internal_state.obs[0]
        
        self.buffer_size = buffer_size
        
        if isinstance(sample_obs, tensordict.TensorDict):
            self.obs = _recursively_create_empty_td(sample_obs, buffer_size)
        else:
            self.obs = torch.empty((buffer_size,) + sample_obs.shape[1:], dtype=sample_obs.dtype, device=sample_obs.device)
        
        action_shape = example_internal_state.action.shape[1:]
        value_shape = example_internal_state.value.shape[1:]
        logprob_shape = example_internal_state.logprob.shape[1:]
        
        self.actions = torch.empty((buffer_size,) + action_shape, dtype=example_internal_state.action.dtype)
        self.logprobs = torch.empty((buffer_size,) + logprob_shape, dtype=example_internal_state.logprob.dtype) 

        self.values = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
        self.last_values = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
        self.advantages = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
        self.returns = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)

        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.next_done = torch.zeros(buffer_size, dtype=torch.bool)
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
        self.obs[current_idx] = internal_state.obs[0]
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
        
    def collate_fn(self, batch: list[tuple]) -> tuple:
        return tuple(torch.stack(items, dim=0) for items in zip(*batch))
    
    def as_dict(self) -> dict:
        idx = self.idx
        obs_slice = self.obs[:idx]
        if isinstance(obs_slice, tensordict.TensorDict):
            obs_slice = obs_slice.to_dict(convert_tensors='numpy')
        else:
            obs_slice = obs_slice.cpu().numpy()
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
            obs_batch_shape=self.obs.batch_size if isinstance(self.obs, tensordict.TensorDict) else self.obs.shape
        )
        return data

    def load_dict(self, data: dict) -> None:
        self.idx = data['idx']
        if isinstance(self.obs, tensordict.TensorDict):
            self.obs[:self.idx] = tensordict.TensorDict(
                data['obs'], batch_size=data['obs_batch_shape']
            )
        else:
            self.obs[:self.idx] = torch.from_numpy(data['obs'])
        self.actions[:self.idx] = torch.from_numpy(data['actions'])
        self.logprobs[:self.idx] = torch.from_numpy(data['logprobs'])
        self.rewards[:self.idx] = torch.from_numpy(data['rewards'])
        self.values[:self.idx] = torch.from_numpy(data['values'])
        self.advantages[:self.idx] = torch.from_numpy(data['advantages'])
        self.returns[:self.idx] = torch.from_numpy(data['returns'])
        self.dones[:self.idx] = torch.from_numpy(data['dones'])
        
        # [WARNING] next_indices is not a tensor, and for some reason it is read-only so we just ignore it here

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
                # check last values not overflow or abs extreme large
                assert abs(next_values).max() < 1e6, f"last_values overflow: {next_values}"
            else:
                next_non_terminal = 1.0 - float(self.dones[next_idx])
                next_values = self.values[next_idx]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * next_gae_lam
            
        self.returns = self.advantages + self.values