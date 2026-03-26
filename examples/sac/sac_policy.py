import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, TypeAlias
from typing_extensions import override
from plugrl_server.policy.base_torch_policy import (
    BaseTorchPolicy,
    BaseTorchPolicyConfig,
)
from plugrl_server.policy.registration import register_policy, register_policy_config

LOG_STD_MAX = 2
LOG_STD_MIN = -5
SACRuntimeState: TypeAlias = dict[str, np.ndarray]


@register_policy_config("sac_policy")
class SACPolicyConfig(BaseTorchPolicyConfig):
    state_dim: int = 11
    action_dim: int = 3
    action_high: float = 1.0
    action_low: float = -1.0
    alpha: float = 0.2
    autotune: bool = True


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, action_high: float, action_low: float
    ):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        self.action_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32
        )
        self.action_bias = torch.tensor((action_high + action_low) / 2.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

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
class SACPolicy(BaseTorchPolicy):
    autotune: bool

    def __init__(self, config: SACPolicyConfig):
        super().__init__(config)
        device = self.device
        actor = Actor(
            config.state_dim, config.action_dim, config.action_high, config.action_low
        ).to(device)
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

    @override
    def prepare_observation(self, _obs: dict[str, Any]) -> np.ndarray:
        state = _obs["states"]["obs"]
        return np.asarray(state, dtype=np.float32)

    @override
    def get_action_and_runtime_state(
        self, _obs: dict[str, Any], random_sample: bool = False
    ) -> tuple[Any, SACRuntimeState]:
        obs = torch.as_tensor(self.prepare_observation(_obs), device=self.device)
        batch_size = obs.shape[0]
        if random_sample:
            action = (
                torch.FloatTensor(obs.shape[0], self.action_dim)
                .uniform_(self.action_low, self.action_high)
                .to(self.device)
            )
        else:
            action, _, _ = self.actor.get_action(obs)
        action_numpy = action.detach().cpu().numpy()
        runtime_state = self.fake_runtime_state(batch_size)
        runtime_state["obs"] = obs.detach().cpu().numpy()
        runtime_state["action"] = action.detach().cpu().numpy()
        return action_numpy[:, None], runtime_state

    @override
    def fake_runtime_state(self, batch_size: int) -> SACRuntimeState:
        return dict(
            obs=np.zeros((batch_size, self.state_dim), dtype=np.float32),
            action=np.zeros((batch_size, self.action_dim), dtype=np.float32),
        )
