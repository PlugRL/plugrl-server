import abc
import dataclasses
import tyro
from typing import Any, Literal
import torch
import torch.nn as nn
import tensordict

from plugrl_server.common.tensor_container import tensor_container, TensorContainer
from plugrl_server.policy.state import (
    PolicyRuntimeState,
)


@dataclasses.dataclass
class BasePolicyConfig:
    algo: tyro.conf._markers.Suppress[str] = "unknown"
    device: torch.device | Literal["cpu", "cuda"] = "cuda"


@tensor_container
class PolicyTensorState(TensorContainer):
    obs: torch.Tensor | tensordict.TensorDict
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


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
    def get_action_and_policy_state(
        self, _obs: dict
    ) -> tuple[Any, PolicyTensorState]: ...

    @abc.abstractmethod
    def fake_policy_state(self, batch_size: int) -> PolicyTensorState: ...

    def get_value(self, _obs: dict) -> torch.Tensor: ...

    def _get_value(self, obs: torch.Tensor | tensordict.TensorDict) -> torch.Tensor: ...

    def _get_action_and_policy_state(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        action: torch.Tensor | None = None,
    ) -> tuple[Any, PolicyTensorState]: ...

__all__ = [
    "BasePolicy",
    "BasePolicyConfig",
    "PolicyTensorState",
    "PolicyRuntimeState",
]
