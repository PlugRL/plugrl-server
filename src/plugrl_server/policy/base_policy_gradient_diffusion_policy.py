import abc
import dataclasses
from collections.abc import Mapping
import numpy as np
import torch
from typing import Any, TypeAlias
from .base_torch_policy import BaseTorchPolicy, BaseTorchPolicyConfig
from .state import NumpyState

TorchTree: TypeAlias = torch.Tensor | dict[str, "TorchTree"]


@dataclasses.dataclass
class DiffusionObs:
    x: torch.Tensor
    t: torch.Tensor
    cond: TorchTree


@dataclasses.dataclass
class DiffusionRuntimeState:
    obs: DiffusionObs
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class BasePolicyGradientDiffusionPolicyConfig(BaseTorchPolicyConfig): ...


class BasePolicyGradientDiffusionPolicy(BaseTorchPolicy):
    actor: torch.nn.Module
    critic: torch.nn.Module
    action_dim: int
    action_horizon: int
    num_denoising_steps: int

    def __init__(self, config: BasePolicyGradientDiffusionPolicyConfig):
        super().__init__(config)

    @abc.abstractmethod
    def _denoising_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree,
        x_next: torch.Tensor | None = None,
        *,
        processed_cond: Any = None,
        sampling_noise_level: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        "return x_next, logprob, entropy"
        ...

    @abc.abstractmethod
    def _get_timesteps(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def _postprocess_action(
        self, action: torch.Tensor, obs: TorchTree
    ) -> Any: ...

    @abc.abstractmethod
    def _initialize_x(self, obs: TorchTree) -> torch.Tensor: ...

    def preprocess_observation(self, obs: TorchTree) -> Any: ...

    def build_model_observation(self, obs: NumpyState) -> TorchTree:
        return _to_torch_tree(obs)

    def get_action_and_runtime_state(
        self, _obs: dict[str, Any], sampling_noise_level: float | None = None
    ) -> tuple[Any, DiffusionRuntimeState]:
        prepared_obs = self.prepare_observation(_obs)
        model_obs = self.build_model_observation(prepared_obs)
        processed_obs = self.preprocess_observation(model_obs)
        timesteps = self._get_timesteps()
        b = _torch_tree_batch_size(model_obs)
        x = self._initialize_x(model_obs)

        runtime_state: DiffusionRuntimeState = self.fake_runtime_state(b)
        for i, t in enumerate(timesteps):
            x_next, logprob, entropy = self._denoising_step(
                x,
                t.repeat(b),
                model_obs,
                processed_cond=processed_obs,
                sampling_noise_level=sampling_noise_level,
            )
            runtime_state.obs.x[:, i] = x
            runtime_state.obs.t[:, i] = t.repeat(b)
            runtime_state.action[:, i] = x_next
            runtime_state.logprob[:, i] = logprob
            runtime_state.entropy[:, i] = entropy
            x = self._iterative_process_action(x_next)

        x = self._postprocess_action(x, model_obs)
        value = self._get_value(model_obs, processed_obs)
        runtime_state.obs.cond = model_obs
        runtime_state.value[:] = value
        return x, runtime_state

    def get_value(self, _obs: dict[str, Any]) -> torch.Tensor:
        prepared_obs = self.prepare_observation(_obs)
        model_obs = self.build_model_observation(prepared_obs)
        processed_obs = self.preprocess_observation(model_obs)
        return self._get_value(model_obs, processed_obs).cpu()

    def _get_value(self, obs: TorchTree, processed_obs: Any = None) -> torch.Tensor: ...

    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> TorchTree: ...

    def fake_runtime_state(self, batch_size: int) -> DiffusionRuntimeState:
        action = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        obs = DiffusionObs(
            x=torch.zeros(
                (
                    batch_size,
                    self.num_denoising_steps,
                    self.action_horizon,
                    self.action_dim,
                )
            ),
            t=torch.zeros((batch_size, self.num_denoising_steps)),
            cond=self.fake_diffusion_cond(batch_size),
        )
        logprob = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        entropy = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        value = torch.zeros((batch_size,))
        return DiffusionRuntimeState(
            obs=obs, action=action, logprob=logprob, entropy=entropy, value=value
        )


def _to_torch_tree(value: NumpyState) -> TorchTree:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Mapping):
        return dict((key, _to_torch_tree(item)) for key, item in value.items())
    raise TypeError(f"Unsupported observation type: {type(value)!r}")


def _torch_tree_batch_size(value: TorchTree) -> int:
    tensor = _first_tensor_in_tree(value)
    if tensor is not None:
        return int(tensor.shape[0])
    raise TypeError("TorchTree observation must expose a batch dimension.")


def _first_tensor_in_tree(value: TorchTree) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    for item in value.values():
        tensor = _first_tensor_in_tree(item)
        if tensor is not None:
            return tensor
    return None
