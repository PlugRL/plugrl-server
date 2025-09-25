import dataclasses
import numpy as np
import torch
import torch.nn as nn

from loguru import logger

from ..base_algorithm import BaseAlgorithm, BaseAlgoConfig
from ..registration import register_algo, register_algo_config
from ...policy.base_policy import BasePolicy, InternalState
from ...buffer.rollout_buffer import RolloutBuffer

UID = "ppo-discrete"

@register_algo_config(UID)
@dataclasses.dataclass
class PPODiscreteAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.99
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1024
    """the total size of the buffer"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 256

@register_algo(UID)
class PPODiscreteAlgorithm(BaseAlgorithm):
    config: PPODiscreteAlgoConfig
    
    def __init__(self, config: PPODiscreteAlgoConfig, policy: BasePolicy):
        super().__init__(config, policy)

        self.policy = policy
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1)
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
    
    def infer(self, obs: dict) -> dict:
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(obs)
        self.rollout_buffer.add_infer(internal_state)
        logger.debug(f"buffer size: {self.rollout_buffer.idx}/{self.rollout_buffer.buffer_size}")
        return {
            "action": action,
        }
        
    def learn(self) -> None:
        dataloader = torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
        ) 
        
        description = self.rollout_buffer.description()
        
        if self.config.anneal_lr:
            frac = 1.0 - (self.rollout_buffer.idx / self.config.buffer_size)
            lrnow = frac * self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrnow

        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), []

        for epoch in range(self.config.update_epochs):
            for batch in dataloader:
                obs, action, oldlogprob, reward, value, advantage, ret = tuple(t.to(self.policy.device) for t in batch)

                _, internal_state = self.policy._get_action_and_internal_state(obs, action)

                internal_state = internal_state.to(self.policy.device)
                newlogprob, entropy, newvalue = internal_state.logprob, internal_state.entropy, internal_state.value
                logratio = newlogprob - oldlogprob
                ratio = logratio.exp()
                 
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                    
                if self.config.norm_adv:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    
                # Policy loss
                pg_loss1 = -advantage * ratio
                pg_loss2 = -advantage * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = value + torch.clamp(newvalue - value, -self.config.clip_coef, self.config.clip_coef)
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()
                    
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break
            
        train_info = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
        }
        train_info.update(description)
        
        train_info_str = "\n".join([f"  {k}: {v:.6f}" for k, v in train_info.items()])

        logger.info(f"PPODiscreteAlgorithm learn info: \n{train_info_str}")

        self.rollout_buffer.reset()

    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        self.rollout_buffer.add_feedback(reward)
        if terminated or truncated:
            with torch.inference_mode():
                last_value = self.policy.get_value(obs)
            self.rollout_buffer.finish_rollout(
                last_value=last_value,
                last_done=terminated or truncated,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                info=info,
            )
        if self.rollout_buffer.full():
            self.learn()