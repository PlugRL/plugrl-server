import dataclasses
import numpy as np
import torch
import torch.nn as nn
import tqdm

from loguru import logger

from vlarl_launcher.common.checkpoint_manager import Checkpoint
from ..base_algorithm import BaseAlgorithm, BaseAlgoConfig
from ..registration import register_algo, register_algo_config
from vlarl_launcher.policy.base_policy import InternalState
from vlarl_launcher.common.checkpoint_manager import Checkpoint
from vlarl_launcher.policy.base_policy_gradient_diffusion_policy import BasePolicyGradientDiffusionPolicy

from .dppo_buffer import DPPOBuffer

try:
    import dppo.util.scheduler as _dppo_scheduler
except ImportError:
    raise ImportError('dppo is not installed. Please install it with pip install "vlarl-infra[dppo]".')

UID = "dppo"

@dataclasses.dataclass
class SchedulerConfig:
    min_lr: float
    warmup_steps: int = 0

@register_algo_config(UID)
@dataclasses.dataclass
class DPPOAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.999
    """total timesteps of the experiments"""
    gamma_denoising: float = 0.99
    """the discount factor for denoising"""
    
    actor_lr: float = 1e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 1e-3
    """the learning rate of the critic optimizer"""
    actor_weight_decay: float = 0.0
    """the weight decay of the actor optimizer"""
    critic_weight_decay: float = 0.0
    """the weight decay of the critic optimizer"""
    actor_lr_scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig(min_lr=1e-4)
    )
    """the learning rate scheduler of the actor optimizer"""
    critic_lr_scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig(min_lr=1e-3)
    )
    """the learning rate scheduler of the critic optimizer"""
    
    buffer_size: int = 20000
    """the total size of the buffer"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
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
    clip_vloss_coef: float | None = None
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float | None = None
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = 1
    """the target KL divergence threshold"""
    
    min_logprob_denoising_std: float = 0.1
    min_sampling_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0
    clip_advantage_upper_quantile: float = 1
    n_critic_warmup_itrs: int = 2

    batch_size: int = 1024
    train_itrs: int = 200
    save_interval: int = 10
    
    @property
    def total_steps(self) -> int:
        return self.buffer_size * self.train_itrs
    
@register_algo(UID)
class DPPOAlgorithm(BaseAlgorithm):
    config: DPPOAlgoConfig
    policy: BasePolicyGradientDiffusionPolicy
    break_action_chunk: bool = False
    
    def __init__(self, config: DPPOAlgoConfig, policy: BasePolicyGradientDiffusionPolicy):
        super().__init__(config, policy)

        self.rollout_buffer = DPPOBuffer(
            buffer_size=config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
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
        if self.policy.critic is not None:
            self.critic_optimizer = torch.optim.AdamW(
                self.policy.critic.parameters(),
                lr=config.critic_lr,
                weight_decay=config.critic_weight_decay,
            )
            self.critic_lr_scheduler = _dppo_scheduler.CosineAnnealingWarmupRestarts(
                self.critic_optimizer,
                first_cycle_steps=self.config.train_itrs,
                max_lr=self.config.critic_lr,
                min_lr=self.config.critic_lr_scheduler.min_lr,
                warmup_steps=self.config.critic_lr_scheduler.warmup_steps
            )
        else:
            self.critic_optimizer = None
            self.critic_lr_scheduler = None
            
        self.save_interval = config.save_interval
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0
            
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
                    "episode/success": info["episode"]["s"],
                }
                logger.info(f"global_step={self.global_step}, " + ", ".join([f"{k}={v}" for k, v in log_dict.items()]))
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1    
        logger.debug(f"Feedback processed. Current buffer size: {self.rollout_buffer.idx}/{self.rollout_buffer.buffer_size}")
        return current_node, self.global_step, log_dict
    
    def learn(self) -> tuple[int, dict]:
        logger.info("Starting learning step")
        logger.info(f"Buffer size: {self.rollout_buffer.idx}/{self.rollout_buffer.buffer_size}")
        self.rollout_buffer.compute_advantages_and_returns()
        logger.info("Computed advantages and returns")
        logger.info("Creating dataloader...")
        dataloader = torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=self.rollout_buffer.collate_fn,
        )
        logger.info("Dataloader created successfully")
        description = self.rollout_buffer.description()
        logger.info("Buffer description computed")
        
        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), []

        max_actor_grad_norms, max_critic_grad_norms = [], []
        
        for update_epoch in range(self.config.update_epochs):
            logger.info(f"Update epoch {update_epoch + 1}/{self.config.update_epochs}")
            break_flag = False
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count == 1:
                    logger.info(f"Processing first batch in epoch {update_epoch + 1}...")
                obs, action, oldlogprob, reward, value, advantage, ret = tuple(t.to(self.policy.device) for t in batch)
                batch_size, ft_denoising_steps = action.shape[:2]
                x, t, cond = obs["x"].reshape(-1, *obs["x"].shape[2:]), obs["t"].reshape(-1), obs["cond"].reshape(-1)
                _, newlogprob, entropy = self.policy._denoising_step(
                    x=x, t=t, cond=cond, x_next=action.reshape(-1, *action.shape[2:]), 
                    min_sampling_denoising_std=self.config.min_logprob_denoising_std
                )
                newlogprob = newlogprob.clamp(min=-5, max=2).mean(dim=(-1, -2)).reshape(batch_size, ft_denoising_steps)
                oldlogprob = oldlogprob.clamp(min=-5, max=2).mean(dim=(-1, -2)).reshape(batch_size, ft_denoising_steps)
                if self.config.norm_adv:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    
                advantage_min = torch.quantile(advantage, self.config.clip_advantage_lower_quantile)
                advantage_max = torch.quantile(advantage, self.config.clip_advantage_upper_quantile)
                advantage = torch.clamp(advantage, advantage_min, advantage_max)
                
                denoising_inds = torch.arange(ft_denoising_steps, device=advantage.device)
                discount = self.config.gamma_denoising ** (ft_denoising_steps - denoising_inds - 1)
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
                # Value loss
                if self.policy.critic is not None:
                    newvalue = self.policy._get_value(obs["cond"][:, 0]).view(-1)
                    if self.config.clip_vloss_coef is not None:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = value + torch.clamp(newvalue - value, -self.config.clip_vloss_coef, self.config.clip_vloss_coef)
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()
                else:
                    v_loss = torch.tensor(0.0)
                    
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss
                
                
                self.actor_optimizer.zero_grad()
                if self.critic_optimizer is not None:
                    self.critic_optimizer.zero_grad()
                loss.backward()

                max_actor_grad_norms.append(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf')).item())
                max_critic_grad_norms.append(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf')).item())
                
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    
                if self.curr_train_itrs >= self.config.n_critic_warmup_itrs:
                    self.actor_optimizer.step()
                
                if self.critic_optimizer is not None:
                    self.critic_optimizer.step()
                    
                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break_flag = True
                    logger.info(f"Early stopping at epoch {update_epoch} due to reaching max KL.")
                    break
                
            if break_flag:
                break
            
        if self.actor_lr_scheduler is not None and self.curr_train_itrs >= self.config.n_critic_warmup_itrs:
            self.actor_lr_scheduler.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()
                
        train_info = {
            "charts/actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "charts/critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"] if self.critic_optimizer is not None else 0.0,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "train/global_step": self.global_step,
            "train/actor_max_grad_norm": max(max_actor_grad_norms) if len(max_actor_grad_norms) > 0 else 0.0,
            "train/critic_max_grad_norm": max(max_critic_grad_norms) if len(max_critic_grad_norms) > 0 else 0.0,
            "train/train_itrs": self.curr_train_itrs,
        }
        train_info.update(description)
        
        train_info_str = "\n".join([f"  {k}: {v:.9f}" for k, v in train_info.items()])

        logger.debug(f"DPPOAlgorithm learn info: \n{train_info_str}")
        
        self.rollout_buffer.reset()
        self.curr_train_itrs += 1
        
        return self.global_step, train_info
    
    def should_learn(self) -> bool:
        return self.rollout_buffer.full()
    
    def should_stop(self) -> bool:
        return self.curr_train_itrs >= self.config.train_itrs
    
    def should_save(self) -> bool:
        return (self.curr_train_itrs % self.save_interval == 0) and (self.curr_train_itrs > self.last_saved_itr)
    
    def create_checkpoint(self) -> Checkpoint:
        self.last_saved_itr = self.curr_train_itrs
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer={
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            },
            meta={
                "train_itrs": self.curr_train_itrs,
                "last_saved_itr": self.last_saved_itr,
            }
        )
        
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.actor_optimizer.load_state_dict(checkpoint.optimizer["actor"])
            if self.critic_optimizer is not None and checkpoint.optimizer["critic"] is not None:
                self.critic_optimizer.load_state_dict(checkpoint.optimizer["critic"])
        if "train_itrs" in checkpoint.meta:
            self.curr_train_itrs = checkpoint.meta["train_itrs"]
        if "last_saved_itr" in checkpoint.meta:
            self.last_saved_itr = checkpoint.meta["last_saved_itr"]
        logger.info(f"Loaded checkpoint at step {self.global_step}, train_itrs {self.curr_train_itrs}, last_saved_itr {self.last_saved_itr}")