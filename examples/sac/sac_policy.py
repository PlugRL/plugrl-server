import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple
from plugrl_server.policy.base_policy import BasePolicy, BasePolicyConfig, InternalState
from plugrl_server.policy.registration import register_policy, register_policy_config

LOG_STD_MAX = 2
LOG_STD_MIN = -5

@register_policy_config("sac_policy")
class SACPolicyConfig(BasePolicyConfig):
    state_dim: int = 11
    action_dim: int = 3
    action_high: float = 1.0
    action_low: float = -1.0
    alpha: float = 0.2
    autotune: bool = True
    
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_high: float, action_low: float):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        self.action_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(
            state_dim + action_dim,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
@register_policy("sac_policy")
class SACPolicy(BasePolicy):
    autotune: bool
    
    def __init__(self, config: SACPolicyConfig):
        super().__init__(config)
        device = self.device
        actor = Actor(config.state_dim, config.action_dim, config.action_high, config.action_low).to(device)
        qf1 = SoftQNetwork(config.state_dim, config.action_dim).to(device)
        qf2 = SoftQNetwork(config.state_dim, config.action_dim).to(device)
        qf1_target = SoftQNetwork(config.state_dim, config.action_dim).to(device)
        qf2_target = SoftQNetwork(config.state_dim, config.action_dim).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = qf1_target
        self.qf2_target = qf2_target
        
        self.autotune = config.autotune
        if self.autotune:
            self.target_entropy = -config.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 0.2  # Fixed alpha value
            
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.alpha = config.alpha
        self.action_high = config.action_high
        self.action_low = config.action_low

    def prepare_observation(self, _obs: dict) -> torch.Tensor:
        state = _obs["states"]["obs"]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state_tensor
    
    def get_action_and_internal_state(self, _obs: dict, random_sample: bool = False) -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        obs = obs.to(self.device)
        batch_size = obs.shape[0]
        if random_sample:
            action = torch.FloatTensor(obs.shape[0], self.action_dim).uniform_(self.action_low, self.action_high).to(self.device)
        else:
            action, _, _ = self.actor.get_action(obs)
        action_numpy = action.detach().cpu().numpy()
        internal_state = self.fake_internal_state(batch_size)
        internal_state.obs = obs
        internal_state.action = action
        return action_numpy[:, None], internal_state.cpu()
        
        
    def fake_internal_state(self, batch_size: int) -> InternalState:
        obs = torch.zeros((batch_size, self.state_dim))
        action = torch.zeros((batch_size, self.action_dim))
        logprob = torch.zeros((batch_size, ))
        entropy = torch.zeros((batch_size, ))
        value = torch.zeros((batch_size, ))
        return InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value)
    