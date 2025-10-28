import abc
import torch
import tensordict
from typing import Any, Literal
from loguru import logger
from .base_policy import BasePolicy, BasePolicyConfig, InternalState

class BaseNFTFlowPolicyConfig(BasePolicyConfig):
    ...

class BaseNFTFlowPolicy(BasePolicy):
    actor: torch.nn.Module
    v_net: torch.nn.Module
    q_net: torch.nn.Module
    action_dim: int
    action_horizon: int
    
    def __init__(self, config: BaseNFTFlowPolicyConfig):
        super().__init__(config)

    @abc.abstractmethod
    def _get_timesteps(self) -> tuple[torch.Tensor, float]:
        ...

    @abc.abstractmethod
    def _initialize_x(self, obs: dict | tensordict.TensorDict) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        ...
        
    @abc.abstractmethod
    def _get_velocity(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        *,
        network_type: Literal["old", "new"] = "new"
    ) -> torch.Tensor:
        ...
        
    @abc.abstractmethod
    def _postprocess_action(self, action: torch.Tensor) -> Any:
        ...

    def _get_action_and_internal_state(self, obs: torch.Tensor | tensordict.TensorDict, action: torch.Tensor | None = None, network_type: Literal["old", "new"] = "new") -> tuple[Any, InternalState]:
        timesteps, dt = self._get_timesteps()
        b = obs.shape[0]
        if action is None:
            x = self._initialize_x(obs)
        else:
            x = action
        
        for t in timesteps:
            v = self._get_velocity(x, t.repeat(b), obs, network_type=network_type)
            x = x + v * dt
            x = self._iterative_process_action(x)

        action = self._postprocess_action(x)
        logprob = torch.zeros(b, device=self.device)
        entropy = torch.zeros(b, device=self.device)
        value = torch.zeros(b, device=self.device)
        return action, InternalState(obs=obs, action=x, logprob=logprob, entropy=entropy, value=value).cpu()

    def get_action_and_internal_state(self, _obs: dict, network_type: Literal["old", "new"] = "new") -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        return self._get_action_and_internal_state(obs, network_type=network_type)

    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        ...

    def fake_internal_state(self, batch_size: int) -> InternalState:
        action = torch.zeros((batch_size, self.action_horizon, self.action_dim))
        obs = self.fake_diffusion_cond(batch_size)
        logprob = torch.zeros((batch_size, ), device=self.device)
        entropy = torch.zeros((batch_size, ), device=self.device)
        value = torch.zeros((batch_size, ), device=self.device)
        return InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value)
    
    def _get_q_value(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        actions: torch.Tensor,
        *,
        network_type: Literal["old", "new"] = "new"
    ) -> torch.Tensor:
        ...
        
    def _get_q_loss(
        self,
        obs: torch.Tensor | tensordict.TensorDict,
        actions: torch.Tensor,
        target_q_values: torch.Tensor,
    ) -> torch.Tensor:
        ...
        
    def soft_update_q_network(self, tau: float) -> None:
        ...
        
    def soft_update_policy_network(self, tau: float) -> None:
        ...
        
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        ...
        
    def get_xt(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...