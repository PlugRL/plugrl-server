import abc
import numpy as np
import dataclasses
from .registration import register_policy_config, register_policy
from .base_policy import BasePolicyConfig, BasePolicy, InternalState

@register_policy_config("dummy-policy")
@dataclasses.dataclass
class DummyPolicyConfig(BasePolicyConfig):
    discrete: bool = True
    action_dim: int = 4

@register_policy("dummy-policy")
class DummyPolicy(BasePolicy):
    discrete: bool
    action_dim: int = 4

    def __init__(self, config: DummyPolicyConfig):
        super().__init__(config)
        self.discrete = config.discrete
        self.action_dim = config.action_dim

    def get_action_and_internal_state(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        if self.discrete:
            action = np.random.randint(0, self.action_dim, size=(1,))
            print(action)
        else:
            action = np.random.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32)
        internal_state = InternalState(obs=obs, action=action)
        return action, internal_state