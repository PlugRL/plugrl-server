import dataclasses
import numpy as np
import torch

from .registration import register_policy_config, register_policy
from .base_torch_policy import BaseTorchPolicyConfig, BaseTorchPolicy
from .state import PolicyRuntimeState


@register_policy_config("dummy-policy")
@dataclasses.dataclass
class DummyPolicyConfig(BaseTorchPolicyConfig):
    discrete: bool = True
    action_dim: int = 4
    action_horizon: int = 4


@register_policy("dummy-policy")
class DummyPolicy(BaseTorchPolicy):
    discrete: bool
    action_dim: int = 4
    action_horizon: int = 4

    def __init__(self, config: DummyPolicyConfig):
        super().__init__(config)
        self.discrete = config.discrete
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

    def get_action_and_runtime_state(
        self, obs: dict, **kwargs
    ) -> tuple[np.ndarray, PolicyRuntimeState]:
        batch_size = self._infer_batch_size(obs)
        if self.discrete:
            action = np.random.randint(
                0, self.action_dim, size=(batch_size, self.action_horizon)
            )
        else:
            action = np.random.uniform(
                -1, 1, size=(batch_size, self.action_horizon, self.action_dim)
            ).astype(np.float32)
        return action, self.fake_runtime_state(batch_size)

    def fake_runtime_state(self, batch_size: int) -> PolicyRuntimeState:
        return dict(
            obs=np.zeros((batch_size,), dtype=np.float32),
            action=np.zeros((batch_size,), dtype=np.float32),
            logprob=np.zeros((batch_size,), dtype=np.float32),
            value=np.zeros((batch_size,), dtype=np.float32),
            entropy=np.zeros((batch_size,), dtype=np.float32),
        )

    def _infer_batch_size(self, obs: dict) -> int:
        if not obs:
            return 1
        first_value = next(iter(obs.values()))
        return len(first_value)

    def _get_value(self, obs: torch.Tensor | dict) -> torch.Tensor:
        # Return a zero value per batch element. Accept raw dict or tensor input.
        try:
            if isinstance(obs, dict):
                batch_size = len(obs.get("text", []))
            else:
                batch_size = obs.shape[0]
        except Exception:
            batch_size = 1
        return torch.zeros((batch_size,), device=self.device)
