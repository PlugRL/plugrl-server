import abc
import dataclasses
import tyro
from typing import Any, Literal
import torch
import torch.nn as nn

@dataclasses.dataclass
class BasePolicyConfig:
    supported_algos: tyro.conf._markers.Suppress[list[str] | None]
    algo: tyro.conf._markers.Suppress[str] = "unknown"
    device: Literal["cpu", "cuda"] = "cuda"
    
@dataclasses.dataclass
class InternalState:
    obs: torch.Tensor 
    action: torch.Tensor 
    logprob: torch.Tensor 
    entropy: torch.Tensor
    value: torch.Tensor
    
    def cpu(self) -> "InternalState":
        return InternalState(
            obs=self.obs.cpu(),
            action=self.action.cpu(),
            logprob=self.logprob.cpu(),
            entropy=self.entropy.cpu(),
            value=self.value.cpu(),
        )
        
    def to(self, device: torch.device) -> "InternalState":
        return InternalState(
            obs=self.obs.to(device),
            action=self.action.to(device),
            logprob=self.logprob.to(device),
            entropy=self.entropy.to(device),
            value=self.value.to(device),
        )

class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, config: BasePolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
        
    @abc.abstractmethod
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