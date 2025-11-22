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
from vlarl_launcher.policy.base_nft_flow_policy import BaseNFTFlowPolicy

from vlarl_launcher.buffer.replay_buffer import ReplayBuffer

UID = "nft"

# @register_algo_config(UID)
@dataclasses.dataclass
class NFTAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.99
    """total timesteps of the experiments"""
    
    buffer_size: int = 1_000_000
    learning_starts: int = 25_000

    batch_size: int = 1024
    tau: float = 5e-3
    actor_tau: float = 5e-3
    beta: float = 1.
    
    total_steps: int = 10000000
    save_interval: int = 1000000
    critic_warmup_steps: int = 100_000
    train_interval: int = 1024
    policy_frequency: int = 2
    nft_train_steps: int = 64
    
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    
    reward_scale: float = 25.0
    reward_bias: float = 0.0
    adv_scale: float = 1.0
    
    adam_betas: tuple = (0.9, 0.999)
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    
@register_algo(UID)
class NFTAlgorithm(BaseAlgorithm):
    config: NFTAlgoConfig
    policy: BaseNFTFlowPolicy
    break_action_chunk: bool = False

    def __init__(self, config: NFTAlgoConfig, policy: BaseNFTFlowPolicy):
        super().__init__(config, policy)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1),
        )
        
        self.q_net_optimizer = torch.optim.Adam(
            params=self.policy.q_net.parameters(),
            lr=self.config.critic_lr,
        )

        self.v_net_optimizer = torch.optim.Adam(
            params=self.policy.v_net.parameters(),
            lr=self.config.critic_lr,
        )
        
        self.actor_optimizer = torch.optim.AdamW(
            params=self.policy.actor.parameters(),
            lr=self.config.actor_lr,
            betas=self.config.adam_betas,
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        self.save_interval = config.save_interval
        self.global_step = 0
        self.last_train_step = 0
        self.last_save_step = 0
        self.train_itr = 0
            
    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(obs, network_type="new")
        return action, internal_state
    
    def feedback(
        self, 
        *, 
        obs: dict, internal_state: InternalState | None, terminated: bool, truncated: bool, 
        next_obs: dict, reward: float, next_terminated: bool, next_truncated: bool, info: dict, prev_node: tuple
    ) -> tuple[tuple, int, dict]:
        if internal_state is None:
            raise NotImplementedError
        else:
            internal_next_obs = self.policy.prepare_observation(next_obs)
            current_node = self.replay_buffer.add_frame(
                prev_node=prev_node,
                obs=internal_state.obs,
                next_obs=internal_next_obs,
                action=internal_state.action,
                reward=(reward - self.config.reward_bias) / self.config.reward_scale,
                done=terminated,
                timeout=truncated
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
            self.global_step += 1    
        return current_node, self.global_step, log_dict
    
    def learn(self) -> tuple[int, dict]:
        logger.info("Starting learning step")
        train_info = {}

        self.train_itr += 1

        data = self.replay_buffer.sample(self.config.batch_size)
        next_obs = data.next_obs
        actions = data.actions
        with torch.no_grad():
            _, next_internal_state = self.policy._get_action_and_internal_state(
                obs=next_obs,
                network_type="old"
            )
            next_actions = next_internal_state.action
            next_q_values = self.policy._get_q_value(
                obs=next_obs,
                actions=next_actions,
                network_type="old"
            )
            rewards = data.rewards.to(next_q_values.device)
            dones = data.dones.float().to(next_q_values.device)
            target_q_values = rewards + self.config.gamma * (1 - dones) * next_q_values

        q_loss = self.policy._get_q_loss(
            obs=data.obs,
            actions=data.actions,
            target_q_values=target_q_values,
        )
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()

        obs = data.obs
        adv = torch.zeros_like(rewards)
        
        all_q_values = []
        all_x0 = []
        
        for _ in range(self.config.nft_train_steps):
            with torch.no_grad():
                _, internal_state = self.policy._get_action_and_internal_state(
                    obs=obs,
                    network_type="new"
                )
            actions = internal_state.action
            with torch.no_grad():
                q_values = self.policy._get_q_value(
                    obs=obs,
                    actions=actions,
                    network_type="new"
                )
            
            all_q_values.append(q_values)
            all_x0.append(actions)
            
        all_values = torch.mean(torch.stack(all_q_values, dim=0), dim=0)
        
        ratio = torch.tensor(0.5).to(adv.device)
        nft_loss, positive_loss, negative_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        
        if self.global_step >= self.config.critic_warmup_steps:
            for i in range(self.config.nft_train_steps):
                x0 = all_x0[i]
                noise = self.policy._initialize_x(obs)
                t = self.policy.sample_timesteps(batch_size=self.config.batch_size)
                xt, v_gt = self.policy.get_xt(x0, noise, t)
                vt_new = self.policy._get_velocity(xt, t, obs, network_type="new")
                with torch.no_grad():
                    vt_old = self.policy._get_velocity(xt, t, obs, network_type="old")
                adv = all_q_values[i] - all_values

                ratio = 0.5 + 0.5 * (adv / self.config.adv_scale).clamp(-1, 1)
                vts_plus = (1 - self.config.beta) * vt_old + self.config.beta * vt_new
                vts_minus  = (1 + self.config.beta) * vt_old - self.config.beta * vt_new
                
                positive_loss = (ratio * ((vts_plus - v_gt) ** 2).mean(dim=[1, 2])).mean()
                negative_loss = ((1 - ratio) * ((vts_minus - v_gt) ** 2).mean(dim=[1, 2])).mean()
                nft_loss = (positive_loss + negative_loss) / self.config.beta
                self.actor_optimizer.zero_grad()
                nft_loss.backward()
                self.actor_optimizer.step()

        values = self.policy._get_value(obs)
        v_loss = nn.functional.mse_loss(values, all_values.detach())
        self.v_net_optimizer.zero_grad()
        v_loss.backward()
        self.v_net_optimizer.step()

        train_info.update({
            "loss/q_loss": q_loss.item(),
            "loss/v_loss": v_loss.item(),
            "loss/nft_loss": nft_loss.item(),
            "loss/positive_loss": positive_loss.item(),
            "loss/negative_loss": negative_loss.item(),
            "metric/value/mean_q": next_q_values.mean().item(),
            "metric/value/min_q": next_q_values.min().item(),
            "metric/value/max_q": next_q_values.max().item(),
            "metric/value/std_q": next_q_values.std().item(),
            "metric/reward/mean": data.rewards.mean().item(),
            "metric/reward/std": data.rewards.std().item(),
            "metric/reward/min": data.rewards.min().item(),
            "metric/reward/max": data.rewards.max().item(),
            "metric/adv/mean": adv.mean().item(),
            "metric/adv/std": adv.std().item(),
            "metric/adv/min": adv.min().item(),
            "metric/adv/max": adv.max().item(),
            "metric/r/mean": ratio.mean().item(),
            "metric/r/std": ratio.std().item(),
            "metric/r/min": ratio.min().item(),
            "metric/r/max": ratio.max().item(),
        })
        
        if self.train_itr % self.config.policy_frequency == 0:
            self.policy.soft_update_policy_network(tau=self.config.actor_tau)
            self.policy.soft_update_q_network(tau=self.config.tau)
        
        self.last_train_step = self.global_step
        return self.global_step, train_info
    
    def should_learn(self) -> bool:
        return len(self.replay_buffer) >= self.config.learning_starts and self.global_step - self.last_train_step >= self.config.train_interval
    
    def should_stop(self) -> bool:
        return self.global_step >= self.config.total_steps
    
    def should_save(self) -> bool:
        return self.global_step - self.last_save_step >= self.config.save_interval
    
    def create_checkpoint(self) -> Checkpoint:
        self.last_save_step = self.global_step
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer={
                "q_net_optimizer": self.q_net_optimizer.state_dict(),
                "v_net_optimizer": self.v_net_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            meta={
                "last_train_step": self.last_train_step,
                "last_save_step": self.last_save_step,
            }
        )
        
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            processed_optimizers = {
                "q_net_optimizer": self.q_net_optimizer,
                "v_net_optimizer": self.v_net_optimizer,
                "actor_optimizer": self.actor_optimizer,
            }
            for key, optimizer_state in checkpoint.optimizer.items():
                if key in processed_optimizers:
                    processed_optimizers[key].load_state_dict(optimizer_state)
        if "last_train_step" in checkpoint.meta:
            self.last_train_step = checkpoint.meta["last_train_step"]
        if "last_save_step" in checkpoint.meta:
            self.last_save_step = checkpoint.meta["last_save_step"]
        logger.info(f"Loaded checkpoint at step {self.global_step}, last_train_step {self.last_train_step}, last_save_step {self.last_save_step}")