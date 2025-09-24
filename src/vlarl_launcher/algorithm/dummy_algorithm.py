import dataclasses
import numpy as np

from .base_algorithm import BaseAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

UID = "dummy"

@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    ...
    
@register_algo(UID)
class DummyAlgorithm(BaseAlgorithm):
    def infer(self, obs: dict) -> dict:
        action, _ = self.policy.get_action_and_internal_state(obs)
        return {"action": action}

    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        pass