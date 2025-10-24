import abc
import torch
import tensordict
from typing import Any
from loguru import logger
from .base_policy import BasePolicy, BasePolicyConfig, InternalState

class BasePGDiffusionPolicyConfig(BasePolicyConfig):
    ...

class BasePGDiffusionPolicy(BasePolicy):
    actor: torch.nn.Module
    critic: torch.nn.Module | None
    action_dim: int
    action_horizon: int
    num_denoising_steps: int
    
    def __init__(self, config: BasePGDiffusionPolicyConfig):
        super().__init__(config)

    @abc.abstractmethod
    def _denoising_step(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        x_next: torch.Tensor | None = None,
        *,
        min_sampling_denoising_std: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        "return x_next, logprob, entropy"
        ...
    
    @abc.abstractmethod
    def _get_timesteps(self) -> torch.Tensor:
        ...
        
    @abc.abstractmethod
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        ...
    
    @abc.abstractmethod
    def _postprocess_action(self, action: torch.Tensor) -> Any:
        ...
    
    @abc.abstractmethod
    def _initialize_x(self, obs: dict | tensordict.TensorDict) -> torch.Tensor:
        ...
    
    def get_action_and_internal_state(self, _obs: dict, min_sampling_denoising_std: float | None = None, **kwargs) -> tuple[Any, InternalState]:
        obs = self.prepare_observation(_obs)
        timesteps = self._get_timesteps()
        b = obs.shape[0]
        x = self._initialize_x(obs)
        
        chain: list[tensordict.TensorDict] = []
        
        for t in timesteps:
            x_next, logprob, entropy = self._denoising_step(x, t.repeat(b), obs, min_sampling_denoising_std=min_sampling_denoising_std)
            chain.append(tensordict.TensorDict(dict(
              obs=dict(x=x, t=t.repeat(b), cond=obs), action=x_next, logprob=logprob, entropy=entropy
            ), batch_size=[b]))
            x = self._iterative_process_action(x_next)

        x = self._postprocess_action(x)
        value = self._get_value(obs)
        chain_tensor: tensordict.TensorDict = tensordict.stack(chain, dim=1)
        obs, action, logprob, entropy = [chain_tensor.get(key) for key in ["obs", "action", "logprob", "entropy"]]
        return x, InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value).cpu()
    
    def get_value(self, _obs: dict) -> torch.Tensor:
        obs = self.prepare_observation(_obs)
        return self._get_value(obs).cpu()
    
    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        ...

    def fake_internal_state(self, batch_size: int) -> InternalState:
        action = torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim))
        obs = tensordict.TensorDict(dict(
            x=torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim)),
            t=torch.zeros((batch_size, self.num_denoising_steps)),
            cond=self.fake_diffusion_cond(batch_size)
        ), batch_size=[batch_size])
        logprob = torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim))
        entropy = torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim))
        value = torch.zeros((batch_size, ))
        return InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value)
