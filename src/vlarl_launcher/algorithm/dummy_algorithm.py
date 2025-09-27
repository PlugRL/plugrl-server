import dataclasses
import numpy as np
import time
from loguru import logger

from .base_algorithm import BaseAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from vlarl_launcher.policy.base_policy import InternalState

UID = "dummy"

@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    fake_inference_duration_sec: float = 0.1
    fake_learn_duration_sec: float = 10.
    fake_learn_freq: int = 100
    
@register_algo(UID)
class DummyAlgorithm(BaseAlgorithm):
    config: DummyAlgoConfig
    
    def __init__(self, config: DummyAlgoConfig, policy):
        super().__init__(config, policy)
        self.counter = 0
    
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        logger.debug(f"DummyAlgorithm.infer called with batch size {len(obs['text'])}")
        action, internal_state = self.policy.get_action_and_internal_state(obs)
        time.sleep(self.config.fake_inference_duration_sec)
        logger.debug(f"DummyAlgorithm.infer returning action shape {action.shape}")
        return action, internal_state

    def feedback(self, *, internal_state: InternalState, terminated: bool, truncated: bool, next_obs: dict, reward: float, info: dict, next_terminated: bool, next_truncated: bool, prev_node: tuple) -> tuple:
        logger.debug(f"DummyAlgorithm.feedback called with prev_node {prev_node}")
        self.counter += 1
        logger.debug(f"Feedback processed. Current counter: {self.counter}")
        return (-1, "")

    def learn(self) -> None:
        logger.debug("DummyAlgorithm.learn called")
        logger.debug(f"Simulating learning for {self.config.fake_learn_duration_sec} seconds...")
        
        time.sleep(self.config.fake_learn_duration_sec)
        self.counter = 0
        
        logger.debug("Learning step completed.")
        
    def should_learn(self) -> bool:
        return self.counter >= self.config.fake_learn_freq
    
    def should_stop(self) -> bool:
        return False
    