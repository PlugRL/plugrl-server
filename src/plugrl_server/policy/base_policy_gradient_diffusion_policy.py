import abc
import torch
import tensordict
from typing import Any
from .base_policy import BasePolicy, BasePolicyConfig, InternalState


class BasePolicyGradientDiffusionPolicyConfig(BasePolicyConfig): ...


class BasePolicyGradientDiffusionPolicy(BasePolicy):
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

    def get_action_and_internal_state(
        self, _obs: dict, sampling_noise_level: float | None = None
    ) -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        processed_obs = self.preprocess_observation(obs)
        timesteps = self._get_timesteps()
        b = obs.shape[0]
        x = self._initialize_x(obs)

        internal_state = self.fake_internal_state(b)
        for i, t in enumerate(timesteps):
            x_next, logprob, entropy = self._denoising_step(
                x,
                t.repeat(b),
                obs,
                processed_cond=processed_obs,
                sampling_noise_level=sampling_noise_level,
            )
            internal_state.obs["x"][:, i] = x
            internal_state.obs["t"][:, i] = t.repeat(b)
            internal_state.action[:, i] = x_next
            internal_state.logprob[:, i] = logprob
            internal_state.entropy[:, i] = entropy
            x = self._iterative_process_action(x_next)

        x = self._postprocess_action(x, obs)
        value = self._get_value(obs, processed_obs)
        internal_state.obs["cond"] = obs
        internal_state.value[:] = value
        return x, internal_state

    def get_value(self, _obs: dict) -> torch.Tensor:
        obs = self.prepare_observation(_obs)
        processed_obs = self.preprocess_observation(obs)
        return self._get_value(obs, processed_obs).cpu()

    def _get_value(
        self, obs: torch.Tensor | tensordict.TensorDict, processed_obs: Any = None
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict: ...

    def fake_internal_state(self, batch_size: int) -> InternalState:
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
        return InternalState(
            obs=obs, action=action, logprob=logprob, entropy=entropy, value=value
        )
