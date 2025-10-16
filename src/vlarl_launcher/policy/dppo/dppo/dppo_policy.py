try:
    import dppo
except ImportError:
    raise ImportError('dppo is not installed. Please install it with pip install "vlarl-infra[dppo]".')

from loguru import logger
import pathlib
import dataclasses
import omegaconf
import hydra
import tensordict
import torch
import numpy as np
from typing import Tuple
from vlarl_launcher.paths import PACKAGE_DIR
from ..base_pg_diffusion_policy import BasePGDiffusionPolicy, BasePGDiffusionPolicyConfig
from ...registration import register_policy, register_policy_config

UID = "dppo-policy"

@dataclasses.dataclass
class DPPOCriticObsConfig:
    mlp_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256, 256])
    activation: str = "Mish"
    residual_style: bool = True

@register_policy_config(UID, supported_algos=[("dppo", "default")])
@dataclasses.dataclass
class DPPOPolicyConfig(BasePGDiffusionPolicyConfig):
    env_type: str = "robomimic"
    env_name: str = "square"
    checkpoint_path: pathlib.Path | None = None
    critic: DPPOCriticObsConfig = dataclasses.field(default_factory=DPPOCriticObsConfig)

@register_policy(UID)
class DPPOPolicy(BasePGDiffusionPolicy):
    config: DPPOPolicyConfig
    obskeys: list[str]
    normalization: dict[str, np.ndarray]
    low_dim_keys: list[str]
    obs_dim: int
    
    def __init__(self, config: DPPOPolicyConfig):
        super().__init__(config)
        
        cfg_path = PACKAGE_DIR / "meta" / "dppo" / "cfg" / self.config.env_type / f"{self.config.env_name}.yaml"
        
        cfg = omegaconf.OmegaConf.load(cfg_path)
        omegaconf.OmegaConf.resolve(cfg)
        
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
            self.actor.load_state_dict(checkpoint["model"])
            logger.info(f"Loaded model weights from {self.config.checkpoint_path}")
        self.low_dim_keys = cfg.low_dim_keys
        self.action_dim = self.actor.action_dim
        self.action_horizon = self.actor.horizon_steps
        self.num_denoising_steps = self.actor.denoising_steps
        
        
        normalization_path = PACKAGE_DIR / "meta" / "dppo" / "asset" / self.config.env_type / self.config.env_name / "normalization.npz"
        self.normalization = np.load(normalization_path)
        
        self.to(self.device)
        
    def _get_timesteps(self) -> torch.Tensor:
        timesteps = list(reversed(range(self.actor.denoising_steps)))
        return torch.tensor(timesteps)
    
    def _initialize_x(self, obs: torch.Tensor | tensordict.TensorDict) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        return x

    def prepare_observation(self, _obs: dict) -> torch.Tensor | tensordict.TensorDict:
        state = _obs["states"]
        state_numpy = np.concatenate([state[key] for key in self.low_dim_keys], axis=-1)
        batch_size = state_numpy.shape[0]
        normalized_state_tensor = 2 * (state_numpy - self.normalization["obs_min"]) / (self.normalization["obs_max"] - self.normalization["obs_min"]) - 1
        return tensordict.TensorDict({"state": torch.tensor(normalized_state_tensor, dtype=torch.float32)}, batch_size=[batch_size])
    
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        return tensordict.TensorDict({"state": torch.zeros(batch_size, self.num_denoising_steps, self.obs_dim)}, batch_size=[batch_size])
    
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action
    
    def _postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        if self.actor.final_action_clip_value is not None:
            action = torch.clamp(action, -self.actor.final_action_clip_value, self.actor.final_action_clip_value)
        action_numpy = action.cpu().numpy()
        unnormalized_action = 0.5 * (action_numpy + 1) * (self.normalization["action_max"] - self.normalization["action_min"]) + self.normalization["action_min"]
        unnormalized_action = np.clip(unnormalized_action, self.normalization["action_min"], self.normalization["action_max"])
        return unnormalized_action
    
    def _denoising_step(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        x_next: torch.Tensor | None = None,
        *,
        min_sampling_denoising_std: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        assert t.shape == (B,)
        assert x.shape == (B, self.action_horizon, self.action_dim)
        
        device = self.actor.betas.device
        t = t.to(device)
        cond = {key: value.to(device) for key, value in cond.items()}
        x = x.to(device)

        mean_logvar: Tuple[torch.Tensor, torch.Tensor] = self.actor.p_mean_var(
            x=x, t=t, cond=cond,
        )
        mean, logvar = mean_logvar
        if min_sampling_denoising_std is not None:
            std = torch.clamp(torch.exp(0.5 * logvar), min=min_sampling_denoising_std)
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
    
    def _get_value(self, obs: tensordict.TensorDict) -> torch.Tensor:
        batch_size = obs.shape[0]
        cond = {k: v for k, v in obs.items()}
        if self.critic is not None:
            value = self.critic(cond).squeeze(-1)
            assert value.shape == (batch_size,)
            return value
        else:
            return torch.zeros(batch_size, device=self.device)