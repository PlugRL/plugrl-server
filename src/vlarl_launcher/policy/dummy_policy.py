import numpy as np
import torch
import dataclasses
from loguru import logger
from .registration import register_policy_config, register_policy
from .base_policy import BasePolicyConfig, BasePolicy, InternalState

@register_policy_config("dummy-policy")
@dataclasses.dataclass
class DummyPolicyConfig(BasePolicyConfig):
    discrete: bool = True
    action_dim: int = 4
    action_horizon: int = 4

@register_policy("dummy-policy")
class DummyPolicy(BasePolicy):
    discrete: bool
    action_dim: int = 4
    action_horizon: int = 4

    def __init__(self, config: DummyPolicyConfig):
        super().__init__(config)
        self.discrete = config.discrete
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

    def get_action_and_internal_state(self, obs: dict, **kwargs) -> tuple[np.ndarray, InternalState]:
        batch_size = len(obs["text"])
        if self.discrete:
            action = np.random.randint(0, self.action_dim, size=(batch_size, self.action_horizon))
        else:
            action = np.random.uniform(-1, 1, size=(batch_size, self.action_horizon, self.action_dim)).astype(np.float32)
        logger.debug(f"DummyPolicy.get_action_and_internal_state called with batch size {batch_size}")
        self.fake_internal_state(batch_size)
        logger.debug(f"DummyPolicy.get_action_and_internal_state returning action shape {action.shape}")
        return action, self.fake_internal_state(batch_size)

    def fake_internal_state(self, batch_size: int) -> InternalState:
        return InternalState(
            obs=torch.zeros((batch_size,)),
            action=torch.zeros((batch_size,)),
            logprob=torch.zeros((batch_size,)),
            value=torch.zeros((batch_size,)),
            entropy=torch.zeros((batch_size,))
        )