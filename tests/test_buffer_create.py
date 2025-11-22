import pickle
import time
import torch
import numpy as np
import tensordict
import sys
import os
import psutil
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss.SSS}|{level}|{message}")

def log_cpu_memory_usage(step, phase="unknown"):
    pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.warning("Process not found for memory logging.")
        return

    memory_info = process.memory_info()
    
    memory_rss_gb = memory_info.rss / (1024 ** 3)  
    memory_vms_gb = memory_info.vms / (1024 ** 3)
    
    system_memory = psutil.virtual_memory()
    system_total_gb = system_memory.total / (1024 ** 3)
    system_available_gb = system_memory.available / (1024 ** 3)

    logger.info(
        f"Step {step} ({phase}): CPU memory (Process) - RSS: {memory_rss_gb:.2f}GB, VMS: {memory_vms_gb:.2f}GB | System - Total: {system_total_gb:.2f}GB, Available: {system_available_gb:.2f}GB"
    )

def _recursively_create_empty_td(template_td, buffer_size):
    new_data = {}
    for key, item in template_td.items():
        if isinstance(item, torch.Tensor):
            feature_shape = item.shape[1:] 
            new_shape = (buffer_size,) + feature_shape
            new_data[key] = torch.empty(new_shape, dtype=item.dtype, device=item.device)
        elif isinstance(item, tensordict.TensorDict):
            new_data[key] = _recursively_create_empty_td(item, buffer_size)
        else:
             new_data[key] = item
             
    return tensordict.TensorDict(new_data, batch_size=[buffer_size])

def _recursively_check_td(td, template_td, current_batch_size):
    for key, item in td.items():
        template_item = template_td[key]
        
        if isinstance(item, torch.Tensor):
            expected_shape = torch.Size([current_batch_size]) + template_item.shape[1:]
            assert item.shape == expected_shape, f"TD key '{key}' shape mismatch: Expected {expected_shape}, got {item.shape}"
            
            assert item.dtype == template_item.dtype, f"TD key '{key}' dtype mismatch: Expected {template_item.dtype}, got {item.dtype}"
            
            test_value = 42.0 + (len(key) if isinstance(key, str) else 0)
            if item.dtype in [torch.float32, torch.float64, torch.float16]:
                item[0] = test_value 
                read_value = item[0].flatten()[0].item()
                assert abs(read_value - test_value) < 1e-5, f"TD key '{key}' writability failed (float). Wrote {test_value}, read {read_value}"
            elif item.dtype == torch.bool:
                item[0] = True
                assert item[0].flatten()[0].item() == True, f"TD key '{key}' writability failed (bool)."
            
        elif isinstance(item, tensordict.TensorDict):
            _recursively_check_td(item, template_item, current_batch_size)

def check_buffer_integrity(buffer_tuple, buffer_size, example_internal_state):
    logger.info("--- Running Integrity Check ---")
    obs, actions, rewards, values = buffer_tuple
    
    try:
        _recursively_check_td(obs, example_internal_state.obs[0], buffer_size)
        logger.success("TensorDict (obs) structure and writability are correct.")
    except AssertionError as e:
        logger.error(f"TensorDict check FAILED: {e}")
        return False

    try:
        expected_action_shape = torch.Size([buffer_size]) + example_internal_state.action.shape[1:]
        assert actions.shape == expected_action_shape, f"Actions shape mismatch: {actions.shape}"
        actions[0] = 99.0 
        assert abs(actions[0].flatten()[0].item() - 99.0) < 1e-5, "Actions writability failed."

        expected_reward_shape = torch.Size([buffer_size])
        assert rewards.shape == expected_reward_shape, f"Rewards shape mismatch: {rewards.shape}"
        assert rewards.dtype == torch.float32, f"Rewards dtype mismatch: {rewards.dtype}"
        rewards[0] = 5.0
        assert abs(rewards[0].item() - 5.0) < 1e-5, "Rewards writability failed."

        expected_value_shape = torch.Size([buffer_size]) + example_internal_state.value.shape[1:]
        assert values.shape == expected_value_shape, f"Values shape mismatch: {values.shape}"
        values[0] = 100.0
        assert abs(values[0].flatten()[0].item() - 100.0) < 1e-5, "Values writability failed."
        
        logger.success("All Tensors (actions, rewards, values) structure and writability are correct.")
        return True

    except AssertionError as e:
        logger.error(f"Tensor check FAILED: {e}")
        return False

def create_buffer_method1(buffer_size, example_internal_state):
    buffer_size = buffer_size
    obs = tensordict.stack([example_internal_state.obs[0]] * buffer_size, dim=0)
    actions = torch.concatenate([example_internal_state.action] * buffer_size, dim=0)
    logprobs = torch.concatenate([example_internal_state.logprob] * buffer_size, dim=0)
    rewards = torch.zeros(buffer_size, dtype=torch.float32)
    values = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
    last_values = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
    next_done = torch.zeros(buffer_size, dtype=torch.bool)
    advantages = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
    returns = torch.concatenate([example_internal_state.value] * buffer_size, dim=0)
    dones = torch.zeros(buffer_size, dtype=torch.bool)
    next_indices = np.zeros(buffer_size, dtype=np.int32)
    
    return obs, actions, rewards, values

def create_buffer_method2(buffer_size, example_internal_state):
    sample_obs = example_internal_state.obs[0]
    
    obs = _recursively_create_empty_td(sample_obs, buffer_size)
    
    action_shape = example_internal_state.action.shape[1:]
    value_shape = example_internal_state.value.shape[1:]
    
    actions = torch.empty((buffer_size,) + action_shape, dtype=example_internal_state.action.dtype)
    logprobs = torch.empty(buffer_size, dtype=example_internal_state.logprob.dtype) 
    
    values = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
    last_values = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
    advantages = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
    returns = torch.empty((buffer_size,) + value_shape, dtype=example_internal_state.value.dtype)
    
    rewards = torch.zeros(buffer_size, dtype=torch.float32)
    next_done = torch.zeros(buffer_size, dtype=torch.bool)
    dones = torch.zeros(buffer_size, dtype=torch.bool)
    next_indices = np.zeros(buffer_size, dtype=np.int32)
    
    return obs, actions, rewards, values

def benchmark_buffer_creation(func, buffer_size, example_internal_state, num_runs=5):
    times = []
    
    func(buffer_size, example_internal_state)
    
    for _ in range(num_runs):
        start_time = time.time()
        log_cpu_memory_usage(0, phase="Benchmark Start")
        func(buffer_size, example_internal_state)
        end_time = time.time()
        log_cpu_memory_usage(0, phase="Benchmark End")
        times.append(end_time - start_time)
        
    avg_time = np.mean(times)
    std_dev = np.std(times)
    
    return avg_time, std_dev

if __name__ == '__main__':
    class MockInternalState:
        def __init__(self):
            self.obs = tensordict.TensorDict({
                "camera": tensordict.TensorDict({
                    "rgb": torch.randn(1, 3, 64, 64),
                    "depth": torch.randn(1, 1, 64, 64)
                }, batch_size=[1]),
                "proprio": torch.randn(1, 12)    
            }, batch_size=[1])
            self.action = torch.randn(1, 4)      
            self.logprob = torch.randn(1)
            self.value = torch.randn(1, 1)       
            
    try:
        example_internal_state = pickle.load(open("tmp/debug_buffer_example_internal_state.pkl", "rb"))
        logger.info("Loaded example_internal_state from file.")
    except FileNotFoundError:
        logger.warning("File not found. Using MockInternalState (Nested TD) for demonstration.")
        example_internal_state = MockInternalState()

    BUFFER_SIZE = 20000
    NUM_RUNS = 5

    logger.info(f"--- Benchmarking Buffer Creation (Size: {BUFFER_SIZE}, Runs: {NUM_RUNS}) ---")
    
    # 运行 Method 2 一次并检查完整性
    optimized_buffer_tuple = create_buffer_method2(BUFFER_SIZE, example_internal_state)
    
    if not check_buffer_integrity(optimized_buffer_tuple, BUFFER_SIZE, example_internal_state):
        sys.exit(1)

    # Method 2 基准测试
    avg_time_2, std_dev_2 = benchmark_buffer_creation(
        create_buffer_method2, BUFFER_SIZE, example_internal_state, NUM_RUNS
    )
    logger.info(f"Method 2 (High-Eff.): Avg Time: {avg_time_2*1000:.3f} ms (Std Dev: {std_dev_2*1000:.3f} ms)")

    # Method 1 基准测试
    avg_time_1, std_dev_1 = benchmark_buffer_creation(
        create_buffer_method1, BUFFER_SIZE, example_internal_state, NUM_RUNS
    )
    logger.info(f"Method 1 (Low-Eff.): Avg Time: {avg_time_1*1000:.3f} ms (Std Dev: {std_dev_1*1000:.3f} ms)")

    speed_up = avg_time_1 / avg_time_2 if avg_time_2 > 0 else float('inf')
    logger.success(f"--- Optimization Speedup: Method 2 is {speed_up:.2f}X faster than Method 1 ---")