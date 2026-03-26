import abc
from collections.abc import Mapping
import numpy as np
import torch
import tensordict
from typing import Any
from plugrl_server.common.tensor_container import TensorContainer, tensor_container
from .base_torch_policy import BaseTorchPolicy, BaseTorchPolicyConfig
from .state import NumpyState, PolicyRuntimeState


@tensor_container
class _DiffusionRuntimeState(TensorContainer):
    obs: torch.Tensor | tensordict.TensorDict
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class BasePolicyGradientDiffusionPolicyConfig(BaseTorchPolicyConfig): ...


class BasePolicyGradientDiffusionPolicy(BaseTorchPolicy):
    actor: torch.nn.Module
    critic: torch.nn.Module | None
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
        cond: dict | tensordict.TensorDict,
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
        self, action: torch.Tensor, obs: torch.Tensor | tensordict.TensorDict
    ) -> Any: ...

    @abc.abstractmethod
    def _initialize_x(self, obs: dict | tensordict.TensorDict) -> torch.Tensor: ...

    def preprocess_observation(
        self, obs: tensordict.TensorDict | torch.Tensor
    ) -> Any: ...

    def build_model_observation(
        self, obs: NumpyState
    ) -> torch.Tensor | tensordict.TensorDict:
        return _to_model_observation(obs)

    def get_action_and_runtime_state(
        self, _obs: dict, sampling_noise_level: float | None = None
    ) -> tuple[Any, PolicyRuntimeState]:
        prepared_obs = self.prepare_observation(_obs)
        model_obs = self.build_model_observation(prepared_obs)
        processed_obs = self.preprocess_observation(model_obs)
        timesteps = self._get_timesteps()
        b = model_obs.shape[0]
        x = self._initialize_x(model_obs)

        runtime_state = self.fake_runtime_state(b)
        for i, t in enumerate(timesteps):
            x_next, logprob, entropy = self._denoising_step(
                x,
                t.repeat(b),
                model_obs,
                processed_cond=processed_obs,
                sampling_noise_level=sampling_noise_level,
            )
            runtime_state.obs["x"][:, i] = x
            runtime_state.obs["t"][:, i] = t.repeat(b)
            runtime_state.action[:, i] = x_next
            runtime_state.logprob[:, i] = logprob
            runtime_state.entropy[:, i] = entropy
            x = self._iterative_process_action(x_next)

        x = self._postprocess_action(x, model_obs)
        value = self._get_value(model_obs, processed_obs)
        runtime_state.obs["cond"] = model_obs
        runtime_state.value[:] = value
        return x, runtime_state

    def get_value(self, _obs: dict) -> torch.Tensor:
        prepared_obs = self.prepare_observation(_obs)
        model_obs = self.build_model_observation(prepared_obs)
        processed_obs = self.preprocess_observation(model_obs)
        return self._get_value(model_obs, processed_obs).cpu()

    def _get_value(
        self, obs: torch.Tensor | tensordict.TensorDict, processed_obs: Any = None
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict: ...

    def fake_runtime_state(self, batch_size: int) -> PolicyRuntimeState:
        action = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        obs = tensordict.TensorDict(
            dict(
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
            ),
            batch_size=[batch_size],
        )
        logprob = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        entropy = torch.zeros(
            (batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)
        )
        value = torch.zeros((batch_size,))
        return _DiffusionRuntimeState(
            obs=obs, action=action, logprob=logprob, entropy=entropy, value=value
        )


def _to_model_observation(value: NumpyState) -> torch.Tensor | tensordict.TensorDict:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Mapping):
        converted = dict(
            (key, _to_model_observation(item)) for key, item in value.items()
        )
        batch_size = _infer_batch_size(converted)
        if batch_size is None:
            raise TypeError("Mapping observation must expose a batch dimension.")
        return tensordict.TensorDict(converted, batch_size=batch_size)
    raise TypeError(f"Unsupported observation type: {type(value)!r}")


def _infer_batch_size(value: dict[str, Any]) -> list[int] | None:
    for item in value.values():
        if isinstance(item, torch.Tensor):
            return list(item.shape[:1])
        if isinstance(item, tensordict.TensorDict):
            return list(item.batch_size)
    return None
