try:
    import dppo
except ImportError:
    raise ImportError(
        'dppo is not installed. Please install it with pip install "plugrl-server[dppo]".'
    )

from loguru import logger
import pathlib
import dataclasses
import omegaconf
import hydra
import tensordict
import torch
import numpy as np
from typing import Tuple, Any
from plugrl_server.paths import PACKAGE_DIR
from ..base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicy,
    BasePolicyGradientDiffusionPolicyConfig,
)
from ..registration import register_policy, register_policy_config

UID = "dppo-policy"


@dataclasses.dataclass
class DPPOCriticObsConfig:
    mlp_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256, 256])
    activation: str = "Mish"
    residual_style: bool = True


@register_policy_config(UID)
@dataclasses.dataclass
class DPPOPolicyConfig(BasePolicyGradientDiffusionPolicyConfig):
    env_type: str = "gym"
    env_name: str = "hopper-medium-v2"
    checkpoint_path: pathlib.Path | None = None
    critic: DPPOCriticObsConfig = dataclasses.field(default_factory=DPPOCriticObsConfig)


@register_policy_config(UID, "hopper")
@dataclasses.dataclass
class DPPOPolicyConfigHopper(DPPOPolicyConfig):
    env_name: str = "hopper-medium-v2"


@register_policy_config(UID, "walker")
@dataclasses.dataclass
class DPPOPolicyConfigWalker(DPPOPolicyConfig):
    env_name: str = "walker2d-medium-v2"


@register_policy_config(UID, "cheetah")
@dataclasses.dataclass
class DPPOPolicyConfigCheetah(DPPOPolicyConfig):
    env_name: str = "halfcheetah-medium-v2"


@register_policy(UID)
class DPPOPolicy(BasePolicyGradientDiffusionPolicy):
    config: DPPOPolicyConfig
    obskeys: list[str]
    normalization: dict[str, np.ndarray]
    low_dim_keys: list[str]
    obs_dim: int

    def __init__(self, config: DPPOPolicyConfig):
        super().__init__(config)

        cfg_path = (
            PACKAGE_DIR
            / "meta"
            / "dppo"
            / "cfg"
            / self.config.env_type
            / f"{self.config.env_name}.yaml"
        )

        cfg = omegaconf.OmegaConf.load(cfg_path)
        omegaconf.OmegaConf.resolve(cfg)
        cfg.model.device = str(self.device)
        self.actor: dppo.diffusion.DiffusionModel = hydra.utils.instantiate(cfg.model)
        self.obs_dim = self.actor.obs_dim
        if isinstance(config.critic, DPPOCriticObsConfig):
            self.critic = dppo.model.critic.CriticObs(
                self.obs_dim,
                mlp_dims=self.config.critic.mlp_dims,
                activation=self.config.critic.activation,
                residual_style=self.config.critic.residual_style,
            )
        else:
            self.critic = None

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
            self.actor.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"Loaded model weights from {self.config.checkpoint_path}")
        self.low_dim_keys = cfg.low_dim_keys
        self.action_dim = self.actor.action_dim
        self.action_horizon = self.actor.horizon_steps
        self.num_denoising_steps = self.actor.denoising_steps

        normalization_path = (
            PACKAGE_DIR
            / "meta"
            / "dppo"
            / "asset"
            / self.config.env_type
            / self.config.env_name
            / "normalization.npz"
        )
        self.normalization = np.load(normalization_path)

        self.to(self.device)

    def _get_timesteps(self) -> torch.Tensor:
        timesteps = list(reversed(range(self.actor.denoising_steps)))
        return torch.tensor(timesteps)

    def _initialize_x(self, obs: torch.Tensor | tensordict.TensorDict) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device
        )
        return x

    def prepare_observation(self, _obs: dict) -> dict[str, np.ndarray]:
        state = _obs["states"]
        state_numpy = np.concatenate([state[key] for key in self.low_dim_keys], axis=-1)
        normalized_state = (
            2
            * (state_numpy - self.normalization["obs_min"])
            / (self.normalization["obs_max"] - self.normalization["obs_min"])
            - 1
        )
        return dict(state=normalized_state.astype(np.float32))

    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        return tensordict.TensorDict(
            {"state": torch.zeros(batch_size, self.obs_dim)}, batch_size=[batch_size]
        )

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(
        self, action: torch.Tensor, obs: tensordict.TensorDict
    ) -> np.ndarray:
        if self.actor.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.actor.final_action_clip_value,
                self.actor.final_action_clip_value,
            )
        action_numpy = action.cpu().numpy()
        unnormalized_action = (
            0.5
            * (action_numpy + 1)
            * (self.normalization["action_max"] - self.normalization["action_min"])
            + self.normalization["action_min"]
        )
        unnormalized_action = np.clip(
            unnormalized_action,
            self.normalization["action_min"],
            self.normalization["action_max"],
        )
        return unnormalized_action

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
        b = x.shape[0]
        assert t.shape == (b,)
        assert x.shape == (b, self.action_horizon, self.action_dim)

        b_cond = next(iter(cond.values())).shape[0]

        device = self.actor.betas.device
        t = t.to(device)
        if b_cond != b:
            assert b == b_cond * self.actor.denoising_steps
            cond = {
                key: value.to(device).repeat_interleave(
                    self.actor.denoising_steps, dim=0
                )
                for key, value in cond.items()
            }
        else:
            cond = {key: value.to(device) for key, value in cond.items()}
        x = x.to(device)

        mean_logvar: Tuple[torch.Tensor, torch.Tensor] = self.actor.p_mean_var(
            x=x,
            t=t.long(),
            cond=cond,
        )
        mean, logvar = mean_logvar
        if sampling_noise_level is not None:
            std = torch.clamp(torch.exp(0.5 * logvar), min=sampling_noise_level)
        else:
            std = torch.exp(0.5 * logvar)

        dist = torch.distributions.Normal(mean, std)

        if x_next is None:
            noise = torch.randn_like(x).clamp_(
                -self.actor.randn_clip_value, self.actor.randn_clip_value
            )
            x_next = mean + std * noise

        logprob = dist.log_prob(x_next)
        entropy = dist.entropy()

        return x_next, logprob, entropy

    def _get_value(
        self, obs: tensordict.TensorDict, processed_obs=None
    ) -> torch.Tensor:
        obs = obs.to(self.device)
        batch_size = obs.shape[0]
        cond = {k: v for k, v in obs.items()}
        if self.critic is not None:
            value = self.critic(cond).squeeze(-1)
            assert value.shape == (batch_size,)
            return value
        else:
            return torch.zeros(batch_size, device=self.device)
