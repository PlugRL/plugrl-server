"""DPPO's Gaussian MLP policy, loadable from the checkpoints DPPO releases.

DPPO (Ren et al. 2024; irom-lab/dppo, MIT) fine-tunes a Gaussian MLP with PPO
as its baseline on robomimic, from pretrained checkpoints it releases. This
is that policy as `model/common/mlp_gaussian.py`'s `Gaussian_MLP` builds it
with `fixed_std` and `learn_fixed_std`, and as `model/common/gaussian.py`'s
`GaussianModel` samples it, built on the `dppo` package's own `ResidualMLP`
and `CriticObs` so that a released checkpoint loads by its own keys:

  mean      tanh(ResidualMLP([obs_dim, 1024, 1024, 1024, horizon x action]))
            with Mish; one standard deviation per action dimension,
            exp(0.5 x clamp(logvar, logvar_min, logvar_max)), repeated over
            the chunk's steps and independent of the observation.
  sample    a normal draw clamped to the mean +- `randn_clip_value`
            deviations; the mean when `deterministic`.
  logprob   the mean over the chunk's elements of each element's log
            density, clamped to [-5, 2], as `PPO_Gaussian` scores it.
  critic    `CriticObs` with a residual [256, 256, 256] Mish MLP.

Observations are scaled to [-1, 1] and actions back to the environment's
units by DPPO's stored `normalization.npz`, as `dppo-policy` does.

One behaviour worth knowing, kept on purpose. The released square checkpoint
carries `network.logvar_max = 0` (a deviation of 1), pretraining's default,
and DPPO loads it non-strictly over the fine-tuning config's 0.2. So DPPO's
fine-tuning actually bounds the deviation at 1.0, not the 0.2 its config and
paper state. This does the same.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dppo.model.common.critic import CriticObs
from dppo.model.common.mlp import ResidualMLP

from plugrl_server.common.logging_utils import get_logger
from plugrl_server.paths import PACKAGE_DIR

from ..base_torch_policy import BaseTorchPolicy, BaseTorchPolicyConfig
from ..registration import register_policy, register_policy_config

logger = get_logger(__name__)

UID = "dppo-gaussian-policy"


@dataclasses.dataclass
class DPPOGaussianRuntimeState:
    obs: torch.Tensor  # (B, obs_dim), scaled to [-1, 1]
    action: torch.Tensor  # (B, horizon x action_dim), the normalised sample
    logprob: torch.Tensor  # (B,)
    value: torch.Tensor  # (B,)


@register_policy_config(UID)
@dataclasses.dataclass
class DPPOGaussianPolicyConfig(BaseTorchPolicyConfig):
    """DPPO's square Gaussian (`cfg/robomimic/finetune/square/ft_ppo_gaussian_mlp.yaml`)."""

    env_type: str = "robomimic"
    env_name: str = "square"
    state_keys: tuple[str, ...] = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    )
    obs_dim: int = 23
    action_dim: int = 7
    horizon_steps: int = 4
    mlp_dims: tuple[int, ...] = (1024, 1024, 1024)
    fixed_std: float = 0.1
    std_min: float = 0.01
    # Replaced by a checkpoint's own `logvar_max`; see the module docstring.
    std_max: float = 0.2
    randn_clip_value: float = 3.0
    logprob_min: float = -5.0
    logprob_max: float = 2.0
    critic_mlp_dims: tuple[int, ...] = (256, 256, 256)
    # DPPO's released pretrained policy; its `model` weights are loaded.
    checkpoint_path: pathlib.Path | None = None
    # Act with the mean: for evaluation. `ppo` refuses it.
    deterministic: bool = False


class _GaussianMLP(nn.Module):
    """The attributes, and so the state-dict keys, of DPPO's `Gaussian_MLP`."""

    def __init__(self, config: DPPOGaussianPolicyConfig) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.horizon_steps = config.horizon_steps
        self.mlp_mean = ResidualMLP(
            [
                config.obs_dim,
                *config.mlp_dims,
                config.action_dim * config.horizon_steps,
            ],
            activation_type="Mish",
            out_activation_type="Identity",
        )
        self.logvar = nn.Parameter(
            torch.full((config.action_dim,), math.log(config.fixed_std**2))
        )
        self.logvar_min = nn.Parameter(
            torch.tensor(math.log(config.std_min**2)), requires_grad=False
        )
        self.logvar_max = nn.Parameter(
            torch.tensor(math.log(config.std_max**2)), requires_grad=False
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.mlp_mean(state))
        logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
        scale = torch.exp(0.5 * logvar).repeat(self.horizon_steps)
        return mean, scale.expand_as(mean)


@register_policy(UID)
class DPPOGaussianPolicy(BaseTorchPolicy):
    config: DPPOGaussianPolicyConfig

    def __init__(self, config: DPPOGaussianPolicyConfig):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.action_horizon = config.horizon_steps
        self.network = _GaussianMLP(config)
        self.critic = CriticObs(
            config.obs_dim,
            mlp_dims=list(config.critic_mlp_dims),
            activation_type="Mish",
            residual_style=True,
        )
        self.normalization = dict(
            np.load(
                PACKAGE_DIR
                / "meta"
                / "dppo"
                / "asset"
                / config.env_type
                / config.env_name
                / "normalization.npz"
            )
        )
        if config.checkpoint_path is not None:
            self._load_released(config.checkpoint_path)
        self.to(self.device)

    def _load_released(self, path: pathlib.Path) -> None:
        """Load a DPPO pretraining checkpoint's `model`, as `GaussianModel` does.

        DPPO's load is non-strict: fine-tuning creates `logvar`, which
        pretraining did not save. Anything else missing is refused rather than
        left at its initialisation.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(checkpoint["model"], strict=False)
        missing = [k for k in missing if not k.startswith("critic.")]
        if unexpected or set(missing) - {"network.logvar"}:
            raise ValueError(
                f"{path} does not fit DPPO's Gaussian MLP: missing "
                f"{sorted(set(missing) - {'network.logvar'})}, unexpected {unexpected}"
            )
        std_max = math.exp(0.5 * float(self.network.logvar_max))
        logger.info(
            f"Loaded the model weights of {path}; the deviation is bounded at "
            f"{std_max:g} by the checkpoint's logvar_max"
        )

    def extract_model_obs_tensor(self, _obs: dict[str, Any]) -> torch.Tensor:
        states = _obs["states"]
        raw = np.concatenate([states[k] for k in self.config.state_keys], axis=-1)
        n = self.normalization
        z = 2 * (raw - n["obs_min"]) / (n["obs_max"] - n["obs_min"]) - 1
        return torch.as_tensor(z, dtype=torch.float32, device=self.device)

    def _logprob(self, dist: torch.distributions.Normal, action: torch.Tensor):
        return (
            dist.log_prob(action)
            .mean(-1)
            .clamp(self.config.logprob_min, self.config.logprob_max)
        )

    def _value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z).squeeze(-1)

    def get_action_and_runtime_state(
        self, _obs: dict[str, Any]
    ) -> tuple[np.ndarray, DPPOGaussianRuntimeState]:
        z = self.extract_model_obs_tensor(_obs)
        mean, scale = self.network(z)
        dist = torch.distributions.Normal(mean, scale)
        if self.config.deterministic:
            sample = mean
        else:
            clip = self.config.randn_clip_value
            sample = torch.clamp(
                dist.sample(), mean - clip * scale, mean + clip * scale
            )
        state = DPPOGaussianRuntimeState(
            obs=z,
            action=sample,
            logprob=self._logprob(dist, sample),
            value=self._value(z),
        )
        batch = sample.shape[0]
        chunk = (
            sample.reshape(batch, self.action_horizon, self.action_dim).cpu().numpy()
        )
        n = self.normalization
        action = (
            0.5 * (chunk + 1) * (n["action_max"] - n["action_min"]) + n["action_min"]
        )
        action = np.clip(action, n["action_min"], n["action_max"])
        return action.astype(np.float32), state

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probability, entropy and value, each (B,), for PPO's loss."""
        mean, scale = self.network(obs)
        dist = torch.distributions.Normal(mean, scale)
        return self._logprob(dist, action), dist.entropy().mean(-1), self._value(obs)

    def get_value(self, _obs: dict[str, Any]) -> torch.Tensor:
        return self._value(self.extract_model_obs_tensor(_obs)).cpu()

    def fake_runtime_state(self, batch_size: int) -> DPPOGaussianRuntimeState:
        flat = self.action_horizon * self.action_dim
        return DPPOGaussianRuntimeState(
            obs=torch.zeros(batch_size, self.config.obs_dim),
            action=torch.zeros(batch_size, flat),
            logprob=torch.zeros(batch_size),
            value=torch.zeros(batch_size),
        )
