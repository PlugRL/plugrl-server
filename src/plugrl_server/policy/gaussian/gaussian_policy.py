"""A Gaussian MLP policy, as CleanRL's `ppo_continuous_action.py` builds it.

The baseline the expressive policies are measured against: a tanh MLP for the
mean, a log standard deviation that does not depend on the observation, and a
separate tanh MLP for the value. Layers are initialised orthogonally at gain
sqrt(2), except the mean's last layer at 0.01 - so the first policy is centred
on zero with unit deviation in every dimension - and the value's at 1.0.

CleanRL does three things in gymnasium wrappers that PlugRL's environment
client does not do, so they are done here instead:

  ClipAction              the environment receives the sample clipped to
                          [-action_clip, action_clip]; the runtime state keeps
                          the unclipped sample, whose density PPO's ratio is.
  NormalizeObservation    running mean and variance, stored as buffers and
                          updated by the algorithm between iterations rather
                          than at every step, so that an iteration's collection
                          and learning see the same normalisation.
  TransformObservation    the normalised observation clipped to +/-obs_clip.

Rewards are normalised by `ppo`'s buffer, which has them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base_torch_policy import BaseTorchPolicy, BaseTorchPolicyConfig
from ..fpo.fpo_policy import _update_running_stats
from ..registration import register_policy, register_policy_config

UID = "gaussian-policy"


def _layer(in_dim: int, out_dim: int, gain: float = float(np.sqrt(2))) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _tanh_mlp(
    in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, *, out_gain: float
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden in hidden_dims:
        layers += [_layer(in_dim, hidden), nn.Tanh()]
        in_dim = hidden
    layers.append(_layer(in_dim, out_dim, out_gain))
    return nn.Sequential(*layers)


@dataclasses.dataclass
class GaussianRuntimeState:
    obs: torch.Tensor  # (B, obs_dim), as the environment sent it
    action: torch.Tensor  # (B, action_dim), the sample before clipping
    logprob: torch.Tensor  # (B,), summed over the action's dimensions
    value: torch.Tensor  # (B,)


@register_policy_config(UID)
@dataclasses.dataclass
class GaussianPolicyConfig(BaseTorchPolicyConfig):
    obs_dim: int = 17
    action_dim: int = 6
    hidden_dims: tuple[int, ...] = (64, 64)
    # The observation's state keys, concatenated in this order, as for
    # fpo-policy. MuJoCo sends one, "obs".
    state_keys: tuple[str, ...] = ("obs",)
    normalize_observations: bool = True
    obs_clip: float = 10.0
    # Keep the observation statistics as loaded, as fpo-policy can.
    freeze_obs_stats: bool = False
    # HalfCheetah, Hopper and Walker2d all act in [-1, 1].
    action_clip: float = 1.0


@register_policy(UID)
class GaussianPolicy(BaseTorchPolicy):
    config: GaussianPolicyConfig

    def __init__(self, config: GaussianPolicyConfig):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.action_horizon = 1
        self.actor_mean = _tanh_mlp(
            config.obs_dim, config.hidden_dims, config.action_dim, out_gain=0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, config.action_dim))
        self.critic = _tanh_mlp(config.obs_dim, config.hidden_dims, 1, out_gain=1.0)
        self.register_buffer("obs_stats_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("obs_stats_mean", torch.zeros(config.obs_dim))
        self.register_buffer("obs_stats_var_sum", torch.zeros(config.obs_dim))
        self.register_buffer("obs_stats_std", torch.ones(config.obs_dim))
        self.to(self.device)

    def extract_model_obs_tensor(self, _obs: dict[str, Any]) -> torch.Tensor:
        states = _obs["states"]
        keys = self.config.state_keys
        missing = [k for k in keys if k not in states]
        if missing:
            raise KeyError(
                f"state keys {missing} are not in the observation, which has "
                f"{sorted(states)}; set GaussianPolicyConfig.state_keys"
            )
        parts = [
            torch.as_tensor(states[k], dtype=torch.float32, device=self.device)
            for k in keys
        ]
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if x.shape[-1] != self.config.obs_dim:
            raise ValueError(
                f"state keys {list(keys)} give {x.shape[-1]} values per "
                f"observation, but obs_dim is {self.config.obs_dim}"
            )
        return x

    def normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_observations:
            return x
        z = (x - self.obs_stats_mean) / self.obs_stats_std
        return z.clamp(-self.config.obs_clip, self.config.obs_clip)

    @torch.no_grad()
    def update_obs_stats(self, x: torch.Tensor) -> None:
        if not self.config.normalize_observations or self.config.freeze_obs_stats:
            return
        count, mean, var_sum, _ = _update_running_stats(
            x=x.to(self.device),
            count=self.obs_stats_count,
            mean=self.obs_stats_mean,
            var_sum=self.obs_stats_var_sum,
        )
        self.obs_stats_count = count
        self.obs_stats_mean = mean
        self.obs_stats_var_sum = var_sum
        # NormalizeObservation's divisor, sqrt(var + 1e-8).
        self.obs_stats_std = torch.sqrt(var_sum / count + 1e-8)

    def _distribution(self, z: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(z)
        return torch.distributions.Normal(mean, self.actor_logstd.expand_as(mean).exp())

    def get_action_and_runtime_state(
        self, _obs: dict[str, Any]
    ) -> tuple[np.ndarray, GaussianRuntimeState]:
        x = self.extract_model_obs_tensor(_obs)
        z = self.normalize_obs(x)
        dist = self._distribution(z)
        sample = dist.sample()
        state = GaussianRuntimeState(
            obs=x,
            action=sample,
            logprob=dist.log_prob(sample).sum(-1),
            value=self.critic(z).squeeze(-1),
        )
        clip = self.config.action_clip
        action = sample.clamp(-clip, clip).cpu().numpy().astype(np.float32)
        return action[:, None, :], state

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probability, entropy and value, each (B,), for PPO's loss."""
        z = self.normalize_obs(obs)
        dist = self._distribution(z)
        return (
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(z).squeeze(-1),
        )

    def get_value(self, _obs: dict[str, Any]) -> torch.Tensor:
        z = self.normalize_obs(self.extract_model_obs_tensor(_obs))
        return self.critic(z).squeeze(-1).cpu()

    def fake_runtime_state(self, batch_size: int) -> GaussianRuntimeState:
        return GaussianRuntimeState(
            obs=torch.zeros(batch_size, self.config.obs_dim),
            action=torch.zeros(batch_size, self.action_dim),
            logprob=torch.zeros(batch_size),
            value=torch.zeros(batch_size),
        )
