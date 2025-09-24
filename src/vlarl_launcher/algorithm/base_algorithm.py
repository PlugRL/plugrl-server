import abc
import dataclasses

from vlarl_launcher.policy.base_policy import BasePolicy

@dataclasses.dataclass
class BaseAlgoConfig:
    ...

class BaseAlgorithm(abc.ABC):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    def __init__(self, config: BaseAlgoConfig, policy: BasePolicy):
        self.config = config
        self.policy = policy
        ...
        
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        ...
        
    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        ...