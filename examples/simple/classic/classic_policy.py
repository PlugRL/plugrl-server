import abc
import dataclasses
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from loguru import logger

from ...registration import register_policy, register_policy_config
from ...base_policy import BasePolicyConfig, BasePolicy, InternalState

UID = "classic-policy"

@register_policy_config(UID)
@dataclasses.dataclass
class ClassicPolicyConfig(BasePolicyConfig):
    n_actions: int = 2
    obs_dim: int = 4
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@register_policy(UID)
class ClassicPolicy(BasePolicy):
    config: ClassicPolicyConfig
    
    def __init__(self, config: ClassicPolicyConfig):
        super().__init__(config)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.config.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.config.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.config.n_actions), std=0.01),
        )
        self.to(self.device)
        
    def prepare_observation(self, _obs: dict):
        obs = torch.as_tensor(_obs["states"]["obs"].copy(), dtype=torch.float32)
        return obs.to(self.device)

    def get_action_and_internal_state(self, _obs: dict, **kwargs) -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        return self._get_action_and_internal_state(obs)

    def _get_action_and_internal_state(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[Any, InternalState]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)

        distribution = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        else:
            action = action.to(self.device)
        logprob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        internal_state = InternalState(
            obs=obs,
            action=action,
            logprob=logprob,
            entropy=entropy,
            value=value
        )

        return action.cpu().numpy()[:, None], internal_state.cpu()
    
    def get_value(self, _obs: dict) -> torch.Tensor:
        obs = self.prepare_observation(_obs)
        return self._get_value(obs)
        
    def _get_value(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.critic(obs)
        return value.cpu()
    
    @torch.inference_mode()
    def fake_internal_state(self, batch_size: int = 1) -> InternalState:
        obs = torch.zeros((batch_size, self.config.obs_dim), dtype=torch.float32).to(self.device)
        _, internal_state = self._get_action_and_internal_state(obs)
        return internal_state