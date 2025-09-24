import abc
import dataclasses
import tyro
from typing import Any

@dataclasses.dataclass
class BasePolicyConfig:
    supported_algos: tyro.conf._markers.Suppress[list[str] | None]
    algo: tyro.conf._markers.Suppress[str] = "unknown"
    
@dataclasses.dataclass
class InternalState:
    obs: Any = None
    action: Any = None
    logprob: Any = None
    entropy: Any = None
    value: Any = None

class BasePolicy(abc.ABC):
    def __init__(self, config: BasePolicyConfig):
        self.config = config
        
    @abc.abstractmethod
    def get_action_and_internal_state(self, obs: dict) -> tuple[Any, InternalState]:
        ...