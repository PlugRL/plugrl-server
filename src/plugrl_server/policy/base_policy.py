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
    PolicyStepState,
    PolicyTrainState,
    to_numpy_state,
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

    def export_runtime_state(self, policy_state: PolicyTensorState) -> PolicyRuntimeState:
        return policy_state

    def export_train_state(self, policy_state: PolicyTensorState) -> PolicyTrainState:
        numpy_state = to_numpy_state(policy_state)
        if numpy_state is None:
            return None
        if not isinstance(numpy_state, dict):
            raise TypeError("Train state export must be mapping-like.")
        return numpy_state

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        if isinstance(runtime_state, PolicyTensorState):
            return self.export_train_state(runtime_state)
        raise TypeError(
            "This policy cannot derive train_state from the provided runtime_state."
        )

    def example_train_state(self, batch_size: int) -> PolicyTrainState:
        return self.export_train_state(self.fake_policy_state(batch_size))

    def build_policy_step_state(
        self,
        policy_state: PolicyTensorState,
        *,
        include_train_state: bool = True,
    ) -> PolicyStepState:
        return PolicyStepState(
            runtime_state=self.export_runtime_state(policy_state),
            train_state=(
                self.export_train_state(policy_state) if include_train_state else None
            ),
        )

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
    "PolicyStepState",
    "PolicyTrainState",
]
