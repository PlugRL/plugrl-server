from __future__ import annotations

import dataclasses
import torch
import numpy as np
import torch.nn as nn
from typing import Any

from ..base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
    BasePolicyGradientFlowPolicyConfig,
)
from ..base_policy_gradient_diffusion_policy import TorchTree
from ..registration import register_policy, register_policy_config

UID = "fpo-policy"


class Mlp(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        *,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowPolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        timestep_embed_dim: int,
        *,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flat_action_dim = action_horizon * action_dim
        self.mlp = Mlp(
            (
                obs_dim + self.flat_action_dim + timestep_embed_dim,
                *hidden_dims,
                self.flat_action_dim,
            )
        )

    def forward(
        self, obs: torch.Tensor, x_t: torch.Tensor, t_embed: torch.Tensor
    ) -> torch.Tensor:
        x_t_flat = x_t.reshape(x_t.shape[0], self.flat_action_dim)
        pred = self.mlp(torch.cat([obs, x_t_flat, t_embed], dim=-1))
        return pred.reshape(x_t.shape[0], self.action_horizon, self.action_dim)


class ValueFunction(nn.Module):
    def __init__(
        self, obs_dim: int, *, hidden_dims: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.mlp = Mlp((obs_dim, *hidden_dims, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).squeeze(-1)


@register_policy_config(UID)
@dataclasses.dataclass
class FPOPolicyConfig(BasePolicyGradientFlowPolicyConfig):
    obs_dim: int = 17
    action_dim: int = 6
    flow_steps: int = 10
    timestep_embed_dim: int = 8
    action_horizon: int = 1
    output_mode: str = "u_but_supervise_as_eps"
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25
    normalize_observations: bool = True
    hidden_dims: tuple[int, ...] = (32, 32, 32, 32)
    value_hidden_dims: tuple[int, ...] = (256, 256, 256, 256, 256)


@register_policy(UID)
class FPOPolicy(BasePolicyGradientFlowPolicy):
    config: FPOPolicyConfig
    output_mode: str
    policy_mlp_output_scale: float

    def __init__(self, config: FPOPolicyConfig):
        super().__init__(config)
        self.actor = FlowPolicyNet(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            timestep_embed_dim=config.timestep_embed_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.critic = ValueFunction(
            obs_dim=config.obs_dim,
            hidden_dims=config.value_hidden_dims,
        ).to(self.device)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_denoising_steps = config.flow_steps
        self.dt = -1.0 / float(config.flow_steps)
        self.output_mode = config.output_mode
        self.policy_mlp_output_scale = config.policy_mlp_output_scale
        self.register_buffer("obs_stats_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "obs_stats_mean", torch.zeros(config.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_stats_var_sum", torch.zeros(config.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_stats_std", torch.ones(config.obs_dim, dtype=torch.float32)
        )

    def _get_timesteps(self) -> torch.Tensor:
        return torch.linspace(
            1.0,
            1.0 / self.num_denoising_steps,
            self.num_denoising_steps,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _update_obs_stats(self, state: torch.Tensor) -> None:
        (
            self.obs_stats_count,
            self.obs_stats_mean,
            self.obs_stats_var_sum,
            self.obs_stats_std,
        ) = _update_running_stats(
            x=state,
            count=self.obs_stats_count,
            mean=self.obs_stats_mean,
            var_sum=self.obs_stats_var_sum,
        )

    def _normalize_state_tensor(
        self, state: torch.Tensor, *, update_stats: bool
    ) -> torch.Tensor:
        if not self.config.normalize_observations:
            return state
        if update_stats:
            self._update_obs_stats(state)
        return _normalize_tensor(state, self.obs_stats_mean, self.obs_stats_std)

    def _normalize_state_numpy(
        self, state: np.ndarray, *, update_stats: bool
    ) -> np.ndarray:
        state_tensor = torch.from_numpy(state).to(self.device)
        normalized = self._normalize_state_tensor(state_tensor, update_stats=update_stats)
        return normalized.detach().cpu().numpy().astype(np.float32)

    def embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] != 1:
            raise ValueError(f"Expected t[...,1], got {tuple(t.shape)}")
        half = self.config.timestep_embed_dim // 2
        freqs = (2.0 ** torch.arange(half, device=t.device, dtype=t.dtype)).view(
            *((1,) * (t.ndim - 1)),
            half,
        )
        scaled = t * freqs
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

    def _initialize_x(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device
        )

    def prepare_observation(self, _obs: dict[str, Any]) -> dict[str, np.ndarray]:
        state = _extract_state_array(_obs)
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        state = state.astype(np.float32)
        if self.config.normalize_observations:
            state = self._normalize_state_numpy(state, update_stats=True)
        return dict(state=state)

    def get_action_and_runtime_state(
        self, _obs: dict[str, Any], sampling_noise_level: float | None = None
    ) -> tuple[Any, Any]:
        action, runtime_state = super().get_action_and_runtime_state(
            _obs, sampling_noise_level=sampling_noise_level
        )
        if self.config.feather_std <= 0:
            return action, runtime_state

        feather_noise = (
            torch.randn_like(runtime_state.action[:, -1]) * self.config.feather_std
        )
        runtime_state.action[:, -1] = runtime_state.action[:, -1] + feather_noise
        return (
            action + feather_noise.detach().cpu().numpy().astype(np.float32),
            runtime_state,
        )

    def fake_diffusion_cond(self, batch_size: int) -> TorchTree:
        return dict(state=torch.zeros(batch_size, self.config.obs_dim))

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(self, action: torch.Tensor, obs: TorchTree) -> np.ndarray:
        return action.detach().cpu().numpy()

    def _predict_v(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree | None,
        *,
        processed_cond: Any = None,
    ) -> torch.Tensor:
        state = processed_cond
        if state is None:
            assert isinstance(cond, dict)
            state = cond["state"].to(self.device)
        if state.shape[0] != x.shape[0]:
            if x.shape[0] % state.shape[0] != 0:
                raise ValueError(
                    f"Cannot repeat FPO cond batch {state.shape[0]} to match x batch {x.shape[0]}."
                )
            state = torch.repeat_interleave(state, x.shape[0] // state.shape[0], dim=0)
        t_embed = self.embed_timestep(t[:, None])
        return self.actor(state, x, t_embed) * self.policy_mlp_output_scale

    def preprocess_observation(self, obs: TorchTree) -> Any:
        assert isinstance(obs, dict)
        return obs["state"].to(self.device)

    def _get_value(self, obs: TorchTree, processed_obs: Any = None) -> torch.Tensor:
        state = processed_obs
        if state is None:
            assert isinstance(obs, dict)
            state = obs["state"].to(self.device)
        return self.critic(state)


def _extract_state_array(obs: dict[str, Any]) -> Any:
    if "state" in obs:
        return obs["state"]
    if "states" in obs:
        states = obs["states"]
        if isinstance(states, dict) and "obs" in states:
            return states["obs"]
        return states
    raise KeyError("FPOPolicy expects observation to contain 'state' or 'states'.")


@torch.no_grad()
def _update_running_stats(
    *,
    x: torch.Tensor,
    count: torch.Tensor,
    mean: torch.Tensor,
    var_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_ndims = x.ndim - mean.ndim
    if batch_ndims < 0:
        raise ValueError(f"x.ndim={x.ndim} must be >= mean.ndim={mean.ndim}")
    batch_axes = tuple(range(batch_ndims))
    batch_count = 1
    for dim in x.shape[:batch_ndims]:
        batch_count *= int(dim)

    new_count = count + torch.as_tensor(
        float(batch_count), device=count.device, dtype=count.dtype
    )
    diff_to_old_mean = x - mean
    new_mean = mean + diff_to_old_mean.sum(dim=batch_axes) / new_count
    new_var_sum = var_sum + (diff_to_old_mean * (x - new_mean)).sum(dim=batch_axes)
    var = new_var_sum / new_count
    std = torch.sqrt(torch.clamp(var, 1e-12, 1e12))
    return new_count, new_mean, new_var_sum, std


def _normalize_tensor(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (x - mean) / std
