import abc
import dataclasses
import numpy as np
import torch
import torch.nn as nn
from typing import Any

from ..registration import register_policy, register_policy_config
from ..base_policy import BasePolicyConfig, BasePolicy, InternalState

UID = "atari-policy"

@register_policy_config(UID, supported_algos=["ppo-discrete"])
@dataclasses.dataclass
class AtariPolicyConfig(BasePolicyConfig):
    n_actions: int = 4
    ...
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@register_policy(UID)
class AtariPolicy(BasePolicy):
    def __init__(self, config: AtariPolicyConfig):
        super().__init__(config)
        self.config = config
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7 * 7 * 64, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, self.config.n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.to(self.device)
        
    def prepare_observation(self, _obs: dict):
        frames = [_obs["images"][f"{i}"][..., 0] for i in range(4)]
        
        # frames [(1, 84, 84), ...] -> (1, *, 84, 84)
        obs = np.stack(frames, axis=1)
        obs = torch.as_tensor(obs, dtype=torch.float32) / 255.0
        return obs.to(self.device)

    def get_action_and_internal_state(self, _obs: dict) -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        return self._get_action_and_internal_state(obs)

    def _get_action_and_internal_state(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[Any, InternalState]:
        hidden = self.network(obs)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        
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

        return action.cpu().numpy(), internal_state.cpu()
    
    def get_value(self, _obs: dict) -> torch.Tensor:
        obs = self.prepare_observation(_obs)
        return self._get_value(obs)
        
    def _get_value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.network(obs)
        value = self.critic(hidden)
        return value.cpu()
    
    @torch.inference_mode()
    def fake_internal_state(self, batch_size: int = 1) -> InternalState:
        obs = torch.zeros((batch_size, 4, 84, 84), dtype=torch.float32).to(self.device)
        _, internal_state = self._get_action_and_internal_state(obs)
        return internal_state