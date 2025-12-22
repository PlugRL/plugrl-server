import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dataclasses
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from loguru import logger

from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.algorithm.base_algorithm import BaseAlgorithm, BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo, register_algo_config
from plugrl_server.policy.base_policy import InternalState
from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.buffer.replay_buffer import ReplayBuffer, ReplayBufferSamples

from sac_policy import SACPolicy, Actor, SoftQNetwork

UID = "sac"

@register_algo_config(UID)
@dataclasses.dataclass
class SACAlgoConfig(BaseAlgoConfig):
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    update_every: int = 10
    update_to_data_ratio: int = 1
    
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    
    policy_frequency: int = 2
    target_network_frequency: int = 1
    
    save_interval: int = 10_000
    
    
@register_algo(UID)
class SACAlgorithm(BaseAlgorithm):
    config: SACAlgoConfig
    policy: SACPolicy
    
    def __init__(self, config: SACAlgoConfig, policy: SACPolicy):
        super().__init__(config=config, policy=policy)
        self.replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            example_internal_state=policy.fake_internal_state(batch_size=1)
        )
        self.save_interval = config.save_interval
        self.global_step = 0
        self.last_learn_step = 0
        self.last_save_step = 0
        self.update_counter = 0

        q_optimizer = torch.optim.Adam(
            list(policy.qf1.parameters()) + list(policy.qf2.parameters()),
            lr=config.q_lr
        )
        actor_optimizer = torch.optim.Adam(
            policy.actor.parameters(),
            lr=config.policy_lr
        )
        if policy.autotune:
            a_optimizer = torch.optim.Adam(
                [policy.log_alpha],
                lr=config.q_lr
            )
        else:
            a_optimizer = None
            
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.a_optimizer = a_optimizer

    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(
                obs, self.global_step < self.config.learning_starts
            )
        return action, internal_state
    
    def feedback(
        self, 
        *, 
        obs: dict, 
        internal_state: InternalState | None, 
        terminated: bool, 
        truncated: bool, 
        next_obs: dict, 
        reward: float, 
        info: dict, 
        next_terminated: bool, 
        next_truncated: bool, 
        prev_node: tuple
    ) -> tuple[tuple, int, dict]:
        assert internal_state is not None, "Internal state must be provided for training."
        current_node = self.replay_buffer.add_frame(
            prev_node=prev_node,
            obs=internal_state.obs,
            next_obs=self.policy.prepare_observation(next_obs),
            action=internal_state.action,
            reward=reward,
            done=next_terminated or next_truncated,
            timeout=False,
        )
        self.global_step += 1
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info:
                log_dict.update({
                    "episode/reward": info["episode"]["r"],
                    "episode/length": info["episode"]["l"],
                    "episode/success": info["episode"]["s"],
                })
                print(f"Episode done at step {self.global_step}: Reward={info['episode']['r']}, Length={info['episode']['l']}, Success={info['episode']['s']}")
        return current_node, self.global_step, log_dict
    
    def update(self, data: ReplayBufferSamples):
        data = data.to(self.policy.device)
        actor: Actor = self.policy.actor
        qf1: SoftQNetwork = self.policy.qf1
        qf2: SoftQNetwork = self.policy.qf2
        qf1_target: SoftQNetwork = self.policy.qf1_target
        qf2_target: SoftQNetwork = self.policy.qf2_target
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
            qf1_next_target = qf1_target(data.next_obs, next_state_actions)
            qf2_next_target = qf2_target(data.next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.policy.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.float().flatten()) * self.config.gamma * (min_qf_next_target).view(-1)
            
        qf1_a_values = qf1(data.obs, data.actions).view(-1)
        qf2_a_values = qf2(data.obs, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        actor_loss, alpha_loss = torch.tensor(0.0), torch.tensor(0.0)
        
        if self.update_counter % self.config.policy_frequency == 0:
            for _ in range(self.config.policy_frequency):
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.policy.alpha * log_pi) - min_qf_pi).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                if self.a_optimizer is not None and self.policy.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    alpha_loss = (-self.policy.log_alpha.exp() * (log_pi + self.policy.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.policy.alpha = self.policy.log_alpha.exp().item()

        if self.update_counter % self.config.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        self.update_counter += 1
        return dict(
            qf1_values=qf1_a_values.mean().item(),
            qf2_values=qf2_a_values.mean().item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss.item(),
            qf_loss=qf_loss.item() / 2
        )
    
    def learn(self) -> tuple[int, dict]:
        self.last_learn_step = self.global_step
        log_dict = {}
        num_updates = int(
            self.config.update_to_data_ratio * self.config.update_every
        )
        update_stats = defaultdict(list)
        for _ in range(num_updates):
            data = self.replay_buffer.sample(self.config.batch_size)
            stats = self.update(data)
            for k, v in stats.items():
                update_stats[k].append(v)
        for k, v in update_stats.items():
            log_dict[f"train/{k}"] = np.mean(v)
            
        log_dict["train/alpha"] = self.policy.alpha
        return self.global_step, log_dict
    
    def should_learn(self) -> bool:
        return (
            self.global_step >= self.config.learning_starts and
            (self.global_step - self.last_learn_step) >= self.config.update_every
        )
        
    def should_stop(self) -> bool:
        return self.global_step >= self.config.total_timesteps
    
    def should_save(self) -> bool:
        return self.global_step - self.last_save_step >= self.save_interval
    
    def create_checkpoint(self) -> Checkpoint:
        self.last_save_step = self.global_step
        optimizers = {
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }
        if self.a_optimizer is not None:
            optimizers["a_optimizer"] = self.a_optimizer.state_dict()
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer=optimizers,
            meta={
                "last_learn_step": self.last_learn_step,
                "last_save_step": self.last_save_step,
                "update_counter": self.update_counter,
            }
        )
    
    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.q_optimizer.load_state_dict(checkpoint.optimizer["q_optimizer"])
            self.actor_optimizer.load_state_dict(checkpoint.optimizer["actor_optimizer"])
            if self.a_optimizer is not None and "a_optimizer" in checkpoint.optimizer:
                self.a_optimizer.load_state_dict(checkpoint.optimizer["a_optimizer"])
        if checkpoint.meta is not None:
            self.last_learn_step = checkpoint.meta.get("last_learn_step", 0)
            self.last_save_step = checkpoint.meta.get("last_save_step", 0)
            self.update_counter = checkpoint.meta.get("update_counter", 0)
        logger.info(f"Loaded checkpoint at step {self.global_step}.")
        
        
if __name__ == "__main__":
    from plugrl_server.cli import main
    main()