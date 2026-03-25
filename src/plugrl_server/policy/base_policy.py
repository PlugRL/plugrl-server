import abc
import dataclasses
import tyro
from typing import Any, Literal
import torch
import torch.nn as nn
import tensordict

from plugrl_server.policy.state import PolicyRuntimeState


@dataclasses.dataclass
class BasePolicyConfig:
    algo: tyro.conf._markers.Suppress[str] = "unknown"
    device: torch.device | Literal["cpu", "cuda"] = "cuda"


class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, config: BasePolicyConfig):
        super().__init__()
        self.config = config
        self.device = (
            config.device
            if isinstance(config.device, torch.device)
            else torch.device(config.device)
        )

    def prepare_observation(
        self, _obs: dict
    ) -> torch.Tensor | tensordict.TensorDict: ...

    @abc.abstractmethod
    def get_action_and_runtime_state(
        self, _obs: dict
    ) -> tuple[Any, PolicyRuntimeState]: ...

    @abc.abstractmethod
    def fake_runtime_state(self, batch_size: int) -> PolicyRuntimeState: ...

    def get_value(self, _obs: dict) -> torch.Tensor: ...

    def _get_value(self, obs: torch.Tensor | tensordict.TensorDict) -> torch.Tensor: ...

    def _get_action_and_runtime_state(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        action: torch.Tensor | None = None,
    ) -> tuple[Any, PolicyRuntimeState]: ...

__all__ = [
    "BasePolicy",
    "BasePolicyConfig",
    "PolicyRuntimeState",
]
