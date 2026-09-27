try:
    import dppo
except ImportError:
    raise ImportError(
        'dppo is not installed. Please install it with pip install "plugrl-server[dppo]".'
    )

import copy
import pathlib
import dataclasses
import omegaconf
import hydra
import torch
import numpy as np
from typing import Tuple, Any
from plugrl_server.paths import PACKAGE_DIR
from plugrl_server.common.data_utils import (
    torch_tree_batch_size,
    torch_tree_repeat_interleave,
    torch_tree_to_device,
)
from plugrl_server.common.logging_utils import get_logger
from ..base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicy,
    BasePolicyGradientDiffusionPolicyConfig,
    TorchTree,
)
from ..registration import register_policy, register_policy_config

logger = get_logger(__name__)

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
    # Fine-tune only the chain's last this-many denoising steps; the earlier
    # ones run a frozen copy of the network as loaded, and only the fine-tuned
    # steps are recorded as PPO samples. DPPO's robomimic fine-tuning uses 10
    # of 20 (`ft_denoising_steps`). None fine-tunes every step.
    ft_denoising_steps: int | None = None
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
    critic: dppo.model.critic.CriticObs

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
        self.critic = dppo.model.critic.CriticObs(
            self.obs_dim,
            mlp_dims=self.config.critic.mlp_dims,
            activation=self.config.critic.activation,
            residual_style=self.config.critic.residual_style,
        )

        if self.config.checkpoint_path is not None:
            # DPPO's pretraining saves `model` and `ema`, and DPPO's own
            # DiffusionModel fine-tunes from `ema` whenever a checkpoint has
            # it; in the released square checkpoint they are 14.6% apart.
            # Strict, because a mismatched checkpoint would otherwise leave
            # the network at its random initialisation without a word.
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location="cpu", weights_only=True
            )
            key = "ema" if "ema" in checkpoint else "model"
            self.actor.load_state_dict(checkpoint[key], strict=True)
            logger.info(f"Loaded the {key} weights of {self.config.checkpoint_path}")
        self.low_dim_keys = cfg.low_dim_keys
        self.action_dim = self.actor.action_dim
        self.action_horizon = self.actor.horizon_steps
        self.num_denoising_steps = self.actor.denoising_steps

        ft = self.config.ft_denoising_steps
        if ft is not None and not 0 < ft <= self.num_denoising_steps:
            raise ValueError(
                f"ft_denoising_steps={ft}, but the chain has "
                f"{self.num_denoising_steps} denoising steps"
            )
        # Taken after the checkpoint is loaded: the early steps stay the
        # pretrained policy's, as in DPPO's VPGDiffusion.
        self.actor_frozen: torch.nn.Module | None = None
        if ft is not None and ft < self.num_denoising_steps:
            self.actor_frozen = copy.deepcopy(self.actor).requires_grad_(False)

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

    @property
    def num_recorded_denoising_steps(self) -> int:
        return self.config.ft_denoising_steps or self.num_denoising_steps

    def _mean_logvar(
        self, x: torch.Tensor, t: torch.Tensor, cond: TorchTree
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDPM step t's mean and log-variance, from the frozen copy for t >= ft."""
        if self.actor_frozen is None:
            return self.actor.p_mean_var(x=x, t=t, cond=cond)
        frozen = t >= self.config.ft_denoising_steps
        if not frozen.any():
            return self.actor.p_mean_var(x=x, t=t, cond=cond)
        with torch.no_grad():
            frozen_mean, frozen_logvar = self.actor_frozen.p_mean_var(
                x=x, t=t, cond=cond
            )
        if frozen.all():
            return frozen_mean, frozen_logvar
        # Neither sampling nor DPPO's loss mixes the two in one batch.
        mean, logvar = self.actor.p_mean_var(x=x, t=t, cond=cond)
        pick = frozen.view(-1, *([1] * (x.dim() - 1)))
        return (
            torch.where(pick, frozen_mean, mean),
            torch.where(pick, frozen_logvar, logvar),
        )

    def _get_timesteps(self) -> torch.Tensor:
        timesteps = list(reversed(range(self.actor.denoising_steps)))
        return torch.tensor(timesteps)

    def _initialize_x(self, batch_size: int) -> torch.Tensor:
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

    def fake_diffusion_cond(self, batch_size: int) -> TorchTree:
        return dict(state=torch.zeros(batch_size, self.obs_dim))

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(self, action: torch.Tensor, obs: TorchTree) -> np.ndarray:
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
        cond: TorchTree,
        x_next: torch.Tensor | None = None,
        *,
        cond_cache: Any = None,
        sampling_noise_level: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = cond_cache
        b = x.shape[0]
        assert t.shape == (b,)
        assert x.shape == (b, self.action_horizon, self.action_dim)

        b_cond = torch_tree_batch_size(cond)

        device = self.actor.betas.device
        t = t.to(device)
        if b_cond != b:
            steps = self.num_recorded_denoising_steps
            assert b == b_cond * steps
            cond = torch_tree_repeat_interleave(
                torch_tree_to_device(cond, device),
                steps,
                dim=0,
            )
        else:
            cond = torch_tree_to_device(cond, device)
        x = x.to(device)

        mean_logvar: Tuple[torch.Tensor, torch.Tensor] = self._mean_logvar(
            x, t.long(), cond
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

    def _get_value(self, obs: TorchTree, obs_cache=None) -> torch.Tensor:
        _ = obs_cache
        cond = torch_tree_to_device(obs, self.device)
        batch_size = torch_tree_batch_size(cond)
        value = self.critic(cond).squeeze(-1)
        assert value.shape == (batch_size,)
        return value
