import abc
import torch
import tensordict
from typing import Any
from loguru import logger
from ..base_policy import BasePolicy, BasePolicyConfig, InternalState

class BaseDPPOPolicyConfig(BasePolicyConfig):
    ...

class BaseDPPOPolicy(BasePolicy):
    action_dim: int
    action_horizon: int
    num_denoising_steps: int
    
    def __init__(self, config: BaseDPPOPolicyConfig):
        super().__init__(config)

    @abc.abstractmethod
    def _denoising_step(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: dict | tensordict.TensorDict, 
        x_next: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        "return x_next, mean, std, logprob, entropy"
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
    
    def get_action_and_internal_state(self, _obs: dict) -> tuple[Any, InternalState]:
        logger.debug(f"DPPOPolicy.get_action_and_internal_state called with keys: {_obs.keys()}")
        obs = self.prepare_observation(_obs)
        logger.debug(f"Prepared observation {obs}")
        timesteps = self._get_timesteps()
        b = obs.shape[0]
        x = self._initialize_x(obs)
        logger.debug(f"Initialized x with shape {x.shape}")
        
        chain: list[tensordict.TensorDict] = []
        
        for t in timesteps:
            x_next, mean, std, logprob, entropy = self._denoising_step(x, t.repeat(b), obs)
            chain.append(tensordict.TensorDict(dict(
              obs=dict(x=x, t=t.repeat(b), cond=obs), action=x_next, logprob=logprob, entropy=entropy
            ), batch_size=[b]))
            x = self._iterative_process_action(x_next)

        logger.debug(f"Denoising chain length: {len(chain)}")
        x = self._postprocess_action(x)
        logger.debug(f"Postprocessed action shape: {x.shape}")
        value = self._get_value(obs)
        
        chain_tensor: tensordict.TensorDict = tensordict.stack(chain, dim=1)
        obs, action, logprob, entropy = [chain_tensor.get(key) for key in ["obs", "action", "logprob", "entropy"]]
        logger.debug(f"""
        DPPOPolicy.get_action_and_internal_state returning action shape {x.shape}
        InternalState obs shape {obs.shape}
        InternalState action shape {action.shape}
        InternalState logprob shape {logprob.shape}
        InternalState entropy shape {entropy.shape}
        InternalState value shape {value.shape}
        """)
        return x, InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value)
    
    @abc.abstractmethod
    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        ...

    def fake_internal_state(self, batch_size: int) -> InternalState:
        action = torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim), device=self.device)
        obs = tensordict.TensorDict(dict(
            x=torch.zeros((batch_size, self.num_denoising_steps, self.action_horizon, self.action_dim), device=self.device),
            t=torch.zeros((batch_size, self.num_denoising_steps), device=self.device),
            cond=self.fake_diffusion_cond(batch_size)
        ), batch_size=[batch_size])
        logprob = torch.zeros((batch_size, self.num_denoising_steps), device=self.device)
        entropy = torch.zeros((batch_size, self.num_denoising_steps), device=self.device)
        value = torch.zeros((batch_size, 1), device=self.device)
        return InternalState(obs=obs, action=action, logprob=logprob, entropy=entropy, value=value)
