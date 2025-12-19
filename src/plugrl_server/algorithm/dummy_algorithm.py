import dataclasses
import numpy as np
import time
import torch

from .base_algorithm import DDPAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from plugrl_server.policy.base_policy import InternalState, BasePolicy
from plugrl_server.common.checkpoint_manager import Checkpoint

UID = "dummy"

@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    fake_inference_duration_sec: float = 0.1
    fake_learn_duration_sec: float = 10.
    fake_learn_freq: int = 100
    
    break_action_chunk: bool = False
    
@register_algo(UID)
class DummyAlgorithm(DDPAlgorithm):
    config: DummyAlgoConfig
    
    def __init__(self, config: DummyAlgoConfig, policy: BasePolicy):
        super().__init__(config, policy)
        self.counter = 0
        self.break_action_chunk = config.break_action_chunk
    
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(obs)
        time.sleep(self.config.fake_inference_duration_sec)
        return action, internal_state

    def feedback(self, *, obs: dict, internal_state: InternalState | None, terminated: bool, truncated: bool, next_obs: dict, reward: float, info: dict, next_terminated: bool, next_truncated: bool, prev_node: tuple) -> tuple:
        self.counter += 1
        return (-1, ""), 0, {}

    def learn(self) -> tuple[int, dict]:
        time.sleep(self.config.fake_learn_duration_sec)
        self.counter = 0
        return 0, {}

    def should_learn(self) -> bool:
        return self.counter >= self.config.fake_learn_freq
    
    def should_stop(self) -> bool:
        return False
    
    def should_save(self) -> bool:
        return False
    
    def create_checkpoint(self) -> Checkpoint:
        return Checkpoint(step=0)
    
    def load_checkpoint(self, checkpoint: Checkpoint):
        pass
    
    def activate_ddp(self, ddp_policy) -> None:
        ...

    def set_device(self, device) -> None:
        ...

    def get_serializable_buffer_data(self) -> dict:
        return {}

    def load_serializable_buffer_data(self, data: dict) -> None:
        ...

    def get_active_policy(self) -> BasePolicy:
        return self.policy

    def load_learner_state(self, checkpoint: Checkpoint) -> None:
        ...
        
    def get_server_data(self) -> tuple[int, dict, dict]:
        return self.global_step, {}, {}

    def load_server_data(self, global_step: int, meta_info: dict, data: dict) -> None:
        self.global_step = global_step
        
    def create_ddp_checkpoint(self) -> Checkpoint:
        return Checkpoint(step=0)