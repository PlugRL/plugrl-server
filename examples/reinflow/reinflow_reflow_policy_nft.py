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
import copy
from typing import Tuple, Literal
from vlarl_launcher.paths import PACKAGE_DIR
from ..base_nft_flow_policy import BaseNFTFlowPolicy, BaseNFTFlowPolicyConfig
from ..registration import register_policy, register_policy_config

UID = "reinflow-r-policy-nft"

@dataclasses.dataclass
class ReinFlowReflowCriticObsConfig:
    mlp_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256, 256])
    activation: str = "Mish"
    residual_style: bool = True

@register_policy_config(UID)
@dataclasses.dataclass
class ReinFlowReflowPolicyNFTConfig(BaseNFTFlowPolicyConfig):
    env_type: str = "gym"
    env_name: str = "hopper-medium-v2"
    
    num_denoising_steps: int = 4
    
    cfg_path: pathlib.Path | None = None
    normalization_path: pathlib.Path | None = None
    checkpoint_path: pathlib.Path | None = None
    critic: ReinFlowReflowCriticObsConfig = dataclasses.field(default_factory=ReinFlowReflowCriticObsConfig)

@register_policy(UID)
class ReinFlowReflowNFTPolicy(BaseNFTFlowPolicy):
    config: ReinFlowReflowPolicyNFTConfig
    obskeys: list[str]
    normalization: dict[str, np.ndarray]
    low_dim_keys: list[str]
    obs_dim: int

    def __init__(self, config: ReinFlowReflowPolicyNFTConfig):
        super().__init__(config)

        if self.config.cfg_path is not None:
            cfg_path = self.config.cfg_path
        else:
            cfg_path = PACKAGE_DIR / "meta" / "reinflow" / "cfg" / self.config.env_type / f"{self.config.env_name}.yaml"

        cfg = omegaconf.OmegaConf.load(cfg_path)
        omegaconf.OmegaConf.resolve(cfg)

        self.actor: reinflow.ReFlow = hydra.utils.instantiate(cfg.model)
        self.actor_old: reinflow.ReFlow = hydra.utils.instantiate(cfg.model)
        self.obs_dim = self.actor.obs_dim
        self.action_dim = self.actor.action_dim
        self.action_horizon = self.actor.horizon_steps
        self.low_dim_keys = cfg.low_dim_keys
        
        self.q_net = reinflow.model.common.critic.CriticObsAct(
            cond_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_steps=self.action_horizon,
            mlp_dims=self.config.critic.mlp_dims,
            activation=self.config.critic.activation,
            residual_style=self.config.critic.residual_style,
        )
        self.q_tgt = copy.deepcopy(self.q_net)
        self.q_tgt.load_state_dict(self.q_net.state_dict())
        
        self.v_net = reinflow.model.common.critic.CriticObs(
            cond_dim=self.obs_dim,
            mlp_dims=self.config.critic.mlp_dims,
            activation=self.config.critic.activation,
            residual_style=self.config.critic.residual_style,
        )
        
        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
            self.actor.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"Loaded model weights from {self.config.checkpoint_path}")
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.num_denoising_steps = self.config.num_denoising_steps
        
        if self.config.normalization_path is not None:
            normalization_path = self.config.normalization_path
        else:
            normalization_path = PACKAGE_DIR / "meta" / "reinflow" / "asset" / self.config.env_type / self.config.env_name / "normalization.npz"
        self.normalization = np.load(normalization_path)
        
        if self.config.normalization_path is not None and self.config.cfg_path is not None:
            logger.warning("Both normalization_path and cfg_path are provided. env_type and env_name will be ignored.")

        self.to(self.device)
        
    def _get_timesteps(self) -> tuple[torch.Tensor, float]:
        timestep = torch.linspace(0, 1 - 1 / self.num_denoising_steps, self.num_denoising_steps)
        return timestep, 1.0 / self.num_denoising_steps

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
        return tensordict.TensorDict({"state": torch.zeros(batch_size, self.obs_dim)}, batch_size=[batch_size])
    
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(*self.actor.act_range)
        return action
    
    def _postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        action_numpy = action.cpu().numpy()
        unnormalized_action = 0.5 * (action_numpy + 1) * (self.normalization["action_max"] - self.normalization["action_min"]) + self.normalization["action_min"]
        unnormalized_action = np.clip(unnormalized_action, self.normalization["action_min"], self.normalization["action_max"])
        return unnormalized_action
    
    def _get_velocity(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        *,
        network_type: Literal["old", "new"] = "new"
    ) -> torch.Tensor:
        B = x.shape[0]
        assert t.shape == (B,)
        assert x.shape == (B, self.action_horizon, self.action_dim)
        
        device = self.device
        t = t.to(device)
        cond = {key: value.to(device) for key, value in cond.items()}
        x = x.to(device)
        if network_type == "old":
            vt = self.actor_old.network(x, t, cond)
        else:
            vt = self.actor.network(x, t, cond)
        return vt
    
    def _get_q_value(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        actions: torch.Tensor,
        *,
        network_type: Literal["old", "new"] = "new"
    ) -> torch.Tensor:
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        if network_type == "old":
            q_values = self.q_tgt(obs, actions)
        else:
            q_values = self.q_net(obs, actions)
        return torch.minimum(q_values[0], q_values[1])
    
    def _get_q_loss(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        actions: torch.Tensor,
        target_q_values: torch.Tensor,
    ) -> torch.Tensor:
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        target_q_values = target_q_values.to(self.device)
        q1, q2 = self.q_net(obs, actions)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q_values)
        q_loss = q1_loss + q2_loss
        return q_loss
    
    def _get_value(self, obs: torch.Tensor | tensordict.TensorDict) -> torch.Tensor:
        if isinstance(obs, tensordict.TensorDict):
            obs_input = {key: value.to(self.device) for key, value in obs.items()}
        else:
            obs_input = obs.to(self.device)
        return self.v_net(obs_input)

    def soft_update_q_network(self, tau: float) -> None:
        for param, target_param in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def soft_update_policy_network(self, tau: float) -> None:
        for param, target_param in zip(self.actor.parameters(), self.actor_old.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, device=self.device)

    def get_xt(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.view(-1, 1, 1).to(self.device)
        x0 = x0.to(self.device)
        noise = noise.to(self.device)
        xt = x0 * t + noise * (1 - t)
        return xt, x0 - noise