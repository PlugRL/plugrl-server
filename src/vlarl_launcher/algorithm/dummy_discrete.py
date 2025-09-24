import dataclasses
import numpy as np

from .base import BaseAlgorithm, BaseAlgoConfig
from vlarl_launcher.common.registration import register_algo, register_algo_config

UID = "dummy-discrete"

@register_algo_config(UID)
@dataclasses.dataclass
class DummyDiscreteAlgoConfig(BaseAlgoConfig):
    action_size: int = 4

@register_algo(UID)
class DummyDiscreteAlgorithm(BaseAlgorithm):
    action_size: int

    def __init__(self, config: DummyDiscreteAlgoConfig):
        self.action_size = config.action_size

    def infer(self, obs: dict) -> dict:
        return {"action": np.random.randint(0, self.action_size, size=(1,)).astype(np.int32)}
    
    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        pass