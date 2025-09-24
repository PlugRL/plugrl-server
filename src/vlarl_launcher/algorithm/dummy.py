import dataclasses
import numpy as np

from .base import BaseAlgorithm, BaseAlgoConfig
from vlarl_launcher.common.registration import register_algo, register_algo_config

UID = "dummy"

@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    action_dim: int = 7

@register_algo(UID)
class DummyAlgorithm(BaseAlgorithm):
    action_dim: int
    
    def __init__(self, config: DummyAlgoConfig):
        self.action_dim = config.action_dim

    def infer(self, obs: dict) -> dict:
        return {"action": np.random.rand(1, self.action_dim).astype(np.float32)}
    
    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        pass