try:
    import reinflow
except ImportError:
    raise ImportError('reinflow is not installed. Please install it with pip install "vlarl-infra[reinflow]".')

from loguru import logger
import pathlib
import dataclasses
import omegaconf
import hydra
import tensordict
import torch
import numpy as np
from typing import Tuple, Any
from vlarl_launcher.paths import PACKAGE_DIR
from ..base_policy_gradient_diffusion_policy import BasePolicyGradientDiffusionPolicy, BasePolicyGradientDiffusionPolicyConfig
from ..registration import register_policy, register_policy_config

UID = "reinflow-r-policy"

@dataclasses.dataclass
class ReinFlowReflowCriticObsConfig:
    mlp_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256, 256])
    activation: str = "Mish"
    residual_style: bool = True

@register_policy_config(UID, supported_algos=[("dummy", "default")])
@dataclasses.dataclass
class ReinFlowReflowPolicyConfig(BasePolicyGradientDiffusionPolicyConfig):
    env_type: str = "gym"
    env_name: str = "hopper-medium-v2"
    
    num_denoising_steps: int = 4
    
    cfg_path: pathlib.Path | None = None
    normalization_path: pathlib.Path | None = None
    checkpoint_path: pathlib.Path | None = None
    critic: ReinFlowReflowCriticObsConfig = dataclasses.field(default_factory=ReinFlowReflowCriticObsConfig)

@register_policy(UID)
class ReinFlowReflowPolicy(BasePolicyGradientDiffusionPolicy):
    config: ReinFlowReflowPolicyConfig
    obskeys: list[str]
    normalization: dict[str, np.ndarray]
    low_dim_keys: list[str]
    obs_dim: int

    def __init__(self, config: ReinFlowReflowPolicyConfig):
        super().__init__(config)

        if self.config.cfg_path is not None:
            cfg_path = self.config.cfg_path
        else:
            cfg_path = PACKAGE_DIR / "meta" / "reinflow" / "cfg" / self.config.env_type / f"{self.config.env_name}.yaml"

        cfg = omegaconf.OmegaConf.load(cfg_path)
        omegaconf.OmegaConf.resolve(cfg)

        self.actor: reinflow.ReFlow = hydra.utils.instantiate(cfg.model)
        self.obs_dim = self.actor.obs_dim
        if isinstance(config.critic, ReinFlowReflowCriticObsConfig):
            self.critic = reinflow.model.common.critic.CriticObs(
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
        self.num_denoising_steps = self.config.num_denoising_steps
        self.dt = 1.0 / self.num_denoising_steps
        
        if self.config.normalization_path is not None:
            normalization_path = self.config.normalization_path
        else:
            normalization_path = PACKAGE_DIR / "meta" / "reinflow" / "asset" / self.config.env_type / self.config.env_name / "normalization.npz"
        self.normalization = np.load(normalization_path)
        
        if self.config.normalization_path is not None and self.config.cfg_path is not None:
            logger.warning("Both normalization_path and cfg_path are provided. env_type and env_name will be ignored.")

        self.to(self.device)
        
    def _get_timesteps(self) -> torch.Tensor:
        timestep = torch.linspace(0, 1 - 1 / self.num_denoising_steps, self.num_denoising_steps)
        return timestep

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
        return tensordict.TensorDict({"state": torch.zeros(batch_size, self.num_denoising_steps, self.obs_dim)}, batch_size=[batch_size, self.num_denoising_steps])
    
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(*self.actor.act_range)
        return action

    def _postprocess_action(self, action: torch.Tensor, obs: tensordict.TensorDict) -> np.ndarray:
        action_numpy = action.cpu().numpy()
        unnormalized_action = 0.5 * (action_numpy + 1) * (self.normalization["action_max"] - self.normalization["action_min"]) + self.normalization["action_min"]
        unnormalized_action = np.clip(unnormalized_action, self.normalization["action_min"], self.normalization["action_max"])
        return unnormalized_action
    
    def get_denoising_logvar(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(1e-5 * torch.ones_like(t).unsqueeze(-1).unsqueeze(-1))
    
    def _denoising_step(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        x_next: torch.Tensor | None = None,
        *,
        processed_cond: Any = None,
        min_sampling_denoising_std: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        assert t.shape == (B,)
        assert x.shape == (B, self.action_horizon, self.action_dim)
        
        device = self.device
        t = t.to(device)
        cond = {key: value.to(device) for key, value in cond.items()}
        x = x.to(device)

        vt = self.actor.network(x, t, cond)
        mean, logvar = x + self.dt * vt, self.get_denoising_logvar(t)
        if min_sampling_denoising_std is not None:
            std = torch.clamp(torch.exp(0.5 * logvar), min=min_sampling_denoising_std)
        else:
            std = torch.exp(0.5 * logvar)
            
        dist = torch.distributions.Normal(mean, std)

        if x_next is None:
            noise = torch.randn_like(x)
            x_next = mean + std * noise

        logprob = dist.log_prob(x_next)
        entropy = dist.entropy()
        
        return x_next, logprob, entropy
    
    def _get_value(self, obs: tensordict.TensorDict) -> torch.Tensor:
        obs = obs.to(self.device)
        batch_size = obs.shape[0]
        cond = {k: v for k, v in obs.items()}
        if self.critic is not None:
            value = self.critic(cond).squeeze(-1)
            assert value.shape == (batch_size,)
            return value
        else:
            return torch.zeros(batch_size, device=self.device)