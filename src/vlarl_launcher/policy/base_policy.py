import abc
import dataclasses
import tyro
from typing import Any, Literal, NamedTuple
import torch
import torch.nn as nn

from vlarl_launcher.common.tensor_container import tensor_container, TensorContainer

@dataclasses.dataclass
class BasePolicyConfig:
    supported_algos: tyro.conf._markers.Suppress[list[tuple[str, str]] | None]
    algo: tyro.conf._markers.Suppress[str] = "unknown"
    device: Literal["cpu", "cuda"] = "cuda"

@tensor_container
class InternalState(TensorContainer):
    obs: torch.Tensor 
    action: torch.Tensor 
    logprob: torch.Tensor 
    entropy: torch.Tensor
    value: torch.Tensor

class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, config: BasePolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
        
    def prepare_observation(self, _obs: dict) -> Any:
        ...
        
    @abc.abstractmethod
    def get_action_and_internal_state(self, _obs: dict) -> tuple[Any, InternalState]:
        ...
        
    @abc.abstractmethod
    def fake_internal_state(self, batch_size: int) -> InternalState:
        ...
        
    def get_value(self, _obs: dict) -> torch.Tensor:
        ...
        
    def _get_action_and_internal_state(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[Any, InternalState]:
        ...