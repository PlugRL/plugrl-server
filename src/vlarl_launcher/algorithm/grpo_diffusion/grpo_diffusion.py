import dataclasses
import numpy as np
import torch
import torch.nn as nn

from loguru import logger

from vlarl_launcher.common.checkpoint_manager import Checkpoint
from ..base_algorithm import BaseAlgorithm, BaseAlgoConfig
from ..registration import register_algo, register_algo_config
from vlarl_launcher.policy.base_policy import InternalState
from vlarl_launcher.common.checkpoint_manager import Checkpoint
from vlarl_launcher.policy.base_policy_gradient_diffusion_policy import BasePolicyGradientDiffusionPolicy

from .grpo_diffusion_buffer import GRPODiffusionBuffer

try:
    import dppo.util.scheduler as _dppo_scheduler
except ImportError:
    raise ImportError('dppo is not installed. Please install it with pip install "vlarl-infra[dppo]".')

UID = "grpo-diffusion"

@dataclasses.dataclass
class SchedulerConfig:
    min_lr: float
    warmup_steps: int = 0

@register_algo_config(UID)
@dataclasses.dataclass
class GRPODiffusionAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.999
    """total timesteps of the experiments"""
    gamma_denoising: float = 0.99
    """the discount factor for denoising"""
    
    group_size: int = 8
    """number of episodes per group for relative advantage computation"""
    
    actor_lr: float = 1e-4
    """the learning rate of the actor optimizer"""
    actor_weight_decay: float = 0.0
    """the weight decay of the actor optimizer"""
    actor_lr_scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig(min_lr=1e-4)
    )
    """the learning rate scheduler of the actor optimizer"""
    
    buffer_size: int = 20000
    """the total size of the buffer"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_ploss_coef: float = 0.01
    """the surrogate clipping coefficient"""
    clip_ploss_coef_base: float = 0.001
    """the base surrogate clipping coefficient"""
    clip_ploss_coef_rate: float = 3
    """the rate of increase of the surrogate clipping coefficient"""
    ent_coef: float = 0.
    """coefficient of the entropy"""
    max_grad_norm: float | None = None
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = 1
    """the target KL divergence threshold"""
    
    min_logprob_denoising_std: float = 0.1
    min_sampling_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0
    clip_advantage_upper_quantile: float = 1

    batch_size: int = 1024
    train_itrs: int = 200
    save_interval: int = 10
    
    @property
    def total_steps(self) -> int:
        return self.buffer_size * self.train_itrs
    
@register_algo(UID)
class GRPODiffusionAlgorithm(BaseAlgorithm):
    config: GRPODiffusionAlgoConfig
    policy: BasePolicyGradientDiffusionPolicy
    break_action_chunk: bool = False
    
    def __init__(self, config: GRPODiffusionAlgoConfig, policy: BasePolicyGradientDiffusionPolicy):
        super().__init__(config, policy)

        self.rollout_buffer = GRPODiffusionBuffer(
            buffer_size=config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1),
            gamma=config.gamma,
            group_size=config.group_size,
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.policy.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        self.actor_lr_scheduler = _dppo_scheduler.CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.config.train_itrs,
            max_lr=self.config.actor_lr,
            min_lr=self.config.actor_lr_scheduler.min_lr,
            warmup_steps=self.config.actor_lr_scheduler.warmup_steps,
        )
            
        self.save_interval = config.save_interval
        self.global_step = 0
        self.curr_train_itrs = 0
            
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(obs, min_sampling_denoising_std=self.config.min_sampling_denoising_std)
        return action, internal_state
    
    def feedback(
        self, 
        *, 
        obs: dict, internal_state: InternalState | None, terminated: bool, truncated: bool, 
        next_obs: dict, reward: float, next_terminated: bool, next_truncated: bool, info: dict, prev_node: tuple
    ) -> tuple[tuple, int, dict]:
        assert internal_state is not None, "Internal state must be provided for feedback."
        current_node = self.rollout_buffer.add_frame(
            prev_node=prev_node,
            internal_state=internal_state,
            reward=reward,
            done=truncated or terminated,
            last_value=None,
            next_done=next_truncated or next_terminated
        )
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info:
                log_dict = {
                    "episode/reward": info["episode"]["r"],
                    "episode/length": info["episode"]["l"],
                    "episode/success": info["episode"]["s"],
                }
                logger.info(f"global_step={self.global_step}, " + ", ".join([f"{k}={v}" for k, v in log_dict.items()]))
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1    
        logger.debug(f"Feedback processed. Current buffer size: {self.rollout_buffer.idx}/{self.rollout_buffer.buffer_size}")
        return current_node, self.global_step, log_dict
    
    def learn(self) -> tuple[int, dict]:
        logger.info("Starting learning step")
        self.rollout_buffer.compute_advantages_and_returns()
        logger.debug("Computed advantages and returns")
        dataloader = torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
        )
        description = self.rollout_buffer.description()
        
        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), []

        max_actor_grad_norms = []
        # import ipdb; ipdb.set_trace()
        for update_epoch in range(self.config.update_epochs):
            break_flag = False
            for batch in dataloader:
                obs, action, oldlogprob, reward, value, advantage, ret = tuple(t.to(self.policy.device) for t in batch)
                
                _, newlogprob, entropy = self.policy._denoising_step(
                    x=obs["x"], t=obs["t"], cond=obs["cond"], x_next=action, 
                    min_sampling_denoising_std=self.config.min_logprob_denoising_std
                )
                newlogprob = newlogprob.clamp(min=-5, max=2).mean(dim=(-1, -2)).view(-1)
                oldlogprob = oldlogprob.clamp(min=-5, max=2).mean(dim=(-1, -2)).view(-1)
                
                if self.config.norm_adv:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    
                advantage_min = torch.quantile(advantage, self.config.clip_advantage_lower_quantile)
                advantage_max = torch.quantile(advantage, self.config.clip_advantage_upper_quantile)
                advantage = torch.clamp(advantage, advantage_min, advantage_max)
                
                # ft_denoising_steps = action.shape[1]
                # discount = torch.tensor([
                #     self.config.gamma_denoising ** (ft_denoising_steps - i - 1) for i in range(ft_denoising_steps)
                # ], device=advantage.device).unsqueeze(0)
                ft_denoising_steps = action.shape[1]
                denoising_inds = torch.arange(ft_denoising_steps, device=advantage.device)
                discount = self.config.gamma_denoising ** (ft_denoising_steps - denoising_inds - 1)
                # advantage shape (batch size, )
                # new advantage shape (batch size, ft_denoising_steps)
                advantage = advantage.unsqueeze(1) * discount.unsqueeze(0)
                
                logratio = newlogprob - oldlogprob
                ratio = logratio.exp()
                
                t = denoising_inds.float() / (ft_denoising_steps - 1)
                if ft_denoising_steps > 1:
                    clip_ploss_coef = self.config.clip_ploss_coef_base + (
                        self.config.clip_ploss_coef - self.config.clip_ploss_coef_base
                    ) * (torch.exp(self.config.clip_ploss_coef_rate * t) - 1) / (
                        np.exp(self.config.clip_ploss_coef_rate) - 1
                    )
                else:
                    clip_ploss_coef = t
                clip_ploss_coef = clip_ploss_coef.unsqueeze(0)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()]
                    
                # Policy loss
                pg_loss1 = -advantage * ratio
                pg_loss2 = -advantage * torch.clamp(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # GRPO does not use value function
                v_loss = torch.tensor(0.0, device=self.policy.device)
                    
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss
                
                
                self.actor_optimizer.zero_grad()
                loss.backward()

                max_actor_grad_norms.append(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf')).item())
                
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    
                self.actor_optimizer.step()
                    
                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break_flag = True
                    logger.info(f"Early stopping at epoch {update_epoch} due to reaching max KL.")
                    break
                
            if break_flag:
                break

        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
                
        train_info = {
            "charts/actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "train/global_step": self.global_step,
            "train/actor_max_grad_norm": max(max_actor_grad_norms) if len(max_actor_grad_norms) > 0 else 0.0,
            "train/train_itrs": self.curr_train_itrs,
        }
        train_info.update(description)
        
        train_info_str = "\n".join([f"  {k}: {v:.9f}" for k, v in train_info.items()])

        logger.debug(f"GRPODiffusionAlgorithm learn info: \n{train_info_str}")
        
        self.rollout_buffer.reset()
        self.curr_train_itrs += 1
        
        return self.global_step, train_info
    
    def should_learn(self) -> bool:
        return self.rollout_buffer.full()
    
    def should_stop(self) -> bool:
        return self.curr_train_itrs >= self.config.train_itrs
    
    def should_save(self) -> bool:
        return (self.curr_train_itrs % self.save_interval == 0) and (self.curr_train_itrs > 0)
    
    def create_checkpoint(self) -> Checkpoint:
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer={
                "actor": self.actor_optimizer.state_dict(),
            },
            meta={
                "train_itrs": self.curr_train_itrs,
            }
        )
        
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.actor_optimizer.load_state_dict(checkpoint.optimizer["actor"])
        if "train_itrs" in checkpoint.meta:
            self.curr_train_itrs = checkpoint.meta["train_itrs"]
        logger.info(f"Loaded checkpoint at step {self.global_step}, train_itrs {self.curr_train_itrs}")