import abc
import dataclasses
import numpy as np

from vlarl_launcher.policy.base_policy import BasePolicy, InternalState
from vlarl_launcher.common.checkpoint_manager import Checkpoint

@dataclasses.dataclass
class BaseAlgoConfig:
    ...

class BaseAlgorithm(abc.ABC):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def __init__(self, config: BaseAlgoConfig, policy: BasePolicy):
        self.config = config
        self.policy = policy
        ...
        
    @abc.abstractmethod
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        ...
    
    @abc.abstractmethod
    def feedback(self, *, internal_state: InternalState, terminated: bool, truncated: bool, next_obs: dict, reward: float, info: dict, next_terminated: bool, next_truncated: bool, prev_node: tuple) -> tuple[tuple, int, dict]:
        ...
        
    @abc.abstractmethod
    def learn(self) -> tuple[int, dict]:
        ...
        
    @abc.abstractmethod
    def should_learn(self) -> bool:
        ...
        
    @abc.abstractmethod
    def should_stop(self) -> bool:
        ...
        
    @abc.abstractmethod
    def should_save(self) -> bool:
        ...
        
    @abc.abstractmethod
    def create_checkpoint(self) -> Checkpoint:
        ...
        
    @abc.abstractmethod
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        ...