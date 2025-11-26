import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from loguru import logger

from vlarl_launcher.common.checkpoint_manager import Checkpoint

from vlarl_launcher.algorithm.base_algorithm import BaseAlgoConfig, DDPAlgorithm
from vlarl_launcher.algorithm.registration import register_algo, register_algo_config
from vlarl_launcher.policy.base_policy import BasePolicy, InternalState
from vlarl_launcher.buffer.rollout_buffer import GAEBuffer
from vlarl_launcher.common.checkpoint_manager import Checkpoint, move_model_to_cpu, move_optimizer_to_cpu

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

    batch_size: int = 256
    total_steps: int = 10000000
    save_interval: int = 1000000

@register_algo(UID)
class PPODiscreteAlgorithm(DDPAlgorithm):
    config: PPODiscreteAlgoConfig
    break_action_chunk: bool = False
    
    def __init__(self, config: PPODiscreteAlgoConfig, policy: BasePolicy):
        super().__init__(config, policy)
        
        self.rollout_buffer: GAEBuffer = GAEBuffer(
            buffer_size=config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        self.optimizer = self.create_optimizer(policy)
        self.sampler = None
        self.global_step = 0
        self.last_save_step = 0
        self.ddp_enabled = False

    def create_optimizer(self, policy: BasePolicy | torch.nn.parallel.DistributedDataParallel) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            policy.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def activate_ddp(self, ddp_policy: torch.nn.parallel.DistributedDataParallel) -> None:
        assert isinstance(ddp_policy.module, BasePolicy), "The DDP policy's module must be an instance of BasePolicy."
        assert dist.is_initialized(), "torch.distributed must be initialized before activating DDP."
        self.policy = ddp_policy
        optimizer_state = self.optimizer.state_dict()
        self.optimizer = self.create_optimizer(ddp_policy)
        self.optimizer.load_state_dict(optimizer_state)
        self.ddp_enabled = True
        # release cuda cache
        torch.cuda.empty_cache()
    
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        with torch.inference_mode():
            if self.ddp_enabled:
                action, internal_state = self.policy.module.get_action_and_internal_state(obs)
            else:
                action, internal_state = self.policy.get_action_and_internal_state(obs)
        return action, internal_state
    
    def pre_learn(self) -> None:
        self.rollout_buffer.compute_advantages_and_returns()
        
    def learn(self) -> tuple[int, dict]:
        if self.ddp_enabled:
            sampler = torch.utils.data.DistributedSampler(
                self.rollout_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )
            batch_size = self.config.batch_size // dist.get_world_size()
        else:
            sampler = None
            batch_size = self.config.batch_size
            
        dataloader = torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
        ) 
        
        description = self.rollout_buffer.description()
        
        if self.config.anneal_lr:
            frac = 1.0 - self.global_step / self.config.total_steps
            lrnow = frac * self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrnow

        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), []

        policy = self.policy.module if self.ddp_enabled else self.policy

        for epoch in range(self.config.update_epochs):
            if sampler is not None and self.ddp_enabled:
                sampler.set_epoch(epoch)
            for batch in dataloader:
                obs, action, oldlogprob, reward, value, advantage, ret = tuple(t.to(policy.device) for t in batch)

                _, internal_state = policy._get_action_and_internal_state(obs, action)

                internal_state = internal_state.to(policy.device)
                newlogprob, entropy, newvalue = internal_state.logprob, internal_state.entropy, internal_state.value
                logratio = newlogprob - oldlogprob
                ratio = logratio.exp()
                 
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                    
                if self.config.norm_adv:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

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
                nn.utils.clip_grad_norm_(policy.parameters(), self.config.max_grad_norm)
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
            "train/global_step": self.global_step,
        }
        train_info.update(description)
        
        return self.global_step, train_info

    def post_learn(self) -> None:
        self.rollout_buffer.reset()
    
    def feedback(
        self, 
        *, 
        obs: dict, internal_state: InternalState | None, terminated: bool, truncated: bool, 
        next_obs: dict, reward: float, next_terminated: bool, next_truncated: bool, info: dict, prev_node: tuple
    ) -> tuple[tuple, int, dict]:
        assert internal_state is not None, "Internal state must be provided for feedback."
        if terminated or truncated:
            with torch.inference_mode():
                last_value = self.policy.get_value(next_obs)
        else:
            last_value = None

        current_node = self.rollout_buffer.add_frame(
            prev_node=prev_node,
            internal_state=internal_state,
            reward=reward,
            done=truncated or terminated,
            last_value=last_value,
            next_done=next_truncated or next_terminated
        )
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info:
                log_dict = {
                    "episode/reward": info["episode"]["r"],
                    "episode/length": info["episode"]["l"],
                }
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1    
        return current_node, self.global_step, log_dict

    def should_learn(self) -> bool:
        return self.rollout_buffer.full()
    
    def should_stop(self) -> bool:
        return self.global_step >= self.config.total_steps
    
    def should_save(self) -> bool:
        return self.global_step - self.last_save_step >= self.config.save_interval
    
    def create_checkpoint(self) -> Checkpoint:
        self.last_save_step = self.global_step
        policy = self.policy.module if self.ddp_enabled else self.policy
        return Checkpoint(
            step=self.global_step,
            model=move_model_to_cpu(policy.state_dict()),
            optimizer=move_optimizer_to_cpu(self.optimizer.state_dict()),
        )
        
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)

        if checkpoint.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer)
        
    def set_device(self, device: torch.device) -> None:
        self.policy.to(device)
        self.policy.device = device

    def get_server_data(self) -> tuple[int, dict, dict]:
        return self.global_step, {}, self.rollout_buffer.as_dict()

    def load_server_data(self, global_step: int, meta_info: dict, data: dict) -> None:
        self.rollout_buffer.load_dict(data)
        self.global_step = global_step

    def load_learner_state(self, checkpoint: Checkpoint) -> None:
        assert checkpoint.model is not None, "Checkpoint model state is None."
        self.policy.load_state_dict(checkpoint.model)
        assert checkpoint.optimizer is not None, "Checkpoint optimizer state is None."
        self.optimizer.load_state_dict(checkpoint.optimizer)