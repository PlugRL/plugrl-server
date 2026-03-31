import dataclasses
import math
import time

import numpy as np
import torch

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.algorithm.registration import register_algo
from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.data_utils import (
    numpy_tree_to_torch,
    torch_tree_get_item,
    torch_tree_to_device,
)
from plugrl_server.policy.base_policy_gradient_diffusion_policy import (
    DiffusionRuntimeState,
)
from plugrl_server.policy.base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
)
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy
from plugrl_server.policy.state import PolicyRuntimeState, PolicyTrainState, to_numpy_state

from .fpo_buffer import FPOBuffer
from .fpo_config import FPOAlgoConfig, UID
from .utils import compute_cfm_loss


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@register_algo(UID)
class FPOAlgorithm(BaseAlgorithm):
    config: FPOAlgoConfig
    policy: BasePolicyGradientFlowPolicy

    def __init__(self, config: FPOAlgoConfig, policy: BasePolicyGradientFlowPolicy):
        super().__init__(config, policy)
        example_train_state = self.example_train_state(batch_size=1)
        if example_train_state is None:
            raise ValueError("FPO requires non-empty train_state for rollout storage.")
        self.rollout_buffer = FPOBuffer(
            buffer_size=config.buffer_size,
            example_train_state=example_train_state,
            n_samples_per_action=config.n_samples_per_action,
            discretize_t_for_training=config.discretize_t_for_training,
            gamma=config.discounting,
            gae_lambda=config.gae_lambda,
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
            lr=config.learning_rate,
        )
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            return self.policy.get_action_and_runtime_state(obs)

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        assert isinstance(runtime_state, DiffusionRuntimeState)
        numpy_state = to_numpy_state(runtime_state)
        if numpy_state is None or not isinstance(numpy_state, dict):
            raise TypeError("FPO requires mapping-like train_state export.")
        action_array = np.asarray(numpy_state["action"], dtype=np.float32)
        value_array = np.asarray(numpy_state["value"], dtype=np.float32)
        final_action = action_array[:, -1]
        value = value_array.reshape(-1, 1)
        return dict(
            obs=np.asarray(numpy_state["obs"]["cond"], dtype=np.float32),
            action=final_action,
            logprob=np.zeros_like(final_action, dtype=np.float32),
            value=value,
        )

    def feedback(
        self,
        *,
        obs: dict,
        runtime_state: PolicyRuntimeState,
        train_state: PolicyTrainState = None,
        terminated: bool,
        truncated: bool,
        next_obs: dict,
        reward: float,
        info: dict,
        next_terminated: bool,
        next_truncated: bool,
        prev_node: tuple,
    ) -> tuple[tuple, int, dict]:
        assert train_state is not None
        reward_value = float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
        current_node = self.rollout_buffer.add_frame(
            prev_node=prev_node,
            train_state=train_state,
            reward=reward_value * self.config.reward_scaling,
            done=terminated or truncated,
            last_value=None,
            next_done=next_terminated or next_truncated,
        )
        self.rollout_buffer.add_next_obs_value_request(obs=next_obs, end_node=current_node)
        if next_terminated or next_truncated:
            if "episode" in info and bool(info["episode"].get("mask", True)):
                self.record_episode_metrics(info["episode"])
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1
        return current_node, self.global_step, dict()

    def pre_learn(self) -> None:
        if isinstance(self.policy, FPOPolicy):
            obs = self.rollout_buffer.train_state_storage.get_item(slice(None))
            obs_torch = torch_tree_to_device(numpy_tree_to_torch(obs), self.policy.device)
            self.policy.update_obs_stats_from_model_obs(obs_torch)
        self.rollout_buffer.compute_advantages_and_returns(
            policy=self.policy,
            batch_size=self.config.batch_size,
        )
        self.rollout_buffer.prepare_fpo_fields(
            self.policy,
            batch_size=self.config.batch_size,
        )

    def get_collect_progress_total(self) -> int | None:
        total_steps = self.get_total_training_steps()
        if total_steps is None:
            return self.config.buffer_size
        remaining = max(0, total_steps - self.global_step)
        return min(self.config.buffer_size, remaining)

    def get_collect_progress_completed(self) -> int | None:
        return len(self.rollout_buffer)

    def get_learn_progress_total(self) -> int | None:
        return self.config.num_updates_per_batch * math.ceil(
            len(self.rollout_buffer) / self.config.batch_size
        )

    def should_learn(self) -> bool:
        if self.rollout_buffer.full():
            return True
        total_steps = self.get_total_training_steps()
        if total_steps is None:
            return False
        return self.global_step >= total_steps and len(self.rollout_buffer) > 0

    def should_stop(self) -> bool:
        assert self.config.global_steps is not None
        return self.global_step >= self.config.global_steps

    def should_save(self) -> bool:
        return self.curr_train_itrs - self.last_saved_itr >= self.config.save_interval

    def create_checkpoint(self) -> Checkpoint:
        self.last_saved_itr = self.curr_train_itrs
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer=self.optimizer.state_dict(),
            meta=dict(curr_train_itrs=self.curr_train_itrs),
        )

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer)
        self.global_step = checkpoint.step
        self.curr_train_itrs = int(checkpoint.meta.get("curr_train_itrs", 0))
        self.last_saved_itr = self.curr_train_itrs

    def _build_train_batch_cache(self) -> tuple:
        idx = len(self.rollout_buffer)
        obs = self.rollout_buffer.train_state_storage.get_item(slice(None, idx))
        obs_torch = torch_tree_to_device(numpy_tree_to_torch(obs), self.policy.device)
        action = torch.from_numpy(self.rollout_buffer.actions[:idx]).to(self.policy.device)
        value = torch.from_numpy(self.rollout_buffer.values[:idx]).to(self.policy.device)
        advantage = torch.from_numpy(self.rollout_buffer.advantages[:idx]).to(
            self.policy.device
        )
        ret = torch.from_numpy(self.rollout_buffer.returns[:idx]).to(self.policy.device)
        loss_eps = torch.from_numpy(self.rollout_buffer.loss_eps[:idx]).to(
            self.policy.device
        )
        loss_t = torch.from_numpy(self.rollout_buffer.loss_t[:idx]).to(self.policy.device)
        initial_cfm_loss = torch.from_numpy(
            self.rollout_buffer.initial_cfm_loss[:idx]
        ).to(self.policy.device)
        _sync_cuda_if_needed(self.policy.device)
        return (
            obs_torch,
            action,
            value,
            advantage,
            ret,
            loss_eps,
            loss_t,
            initial_cfm_loss,
        )

    def _compute_loss(
        self,
        obs,
        action,
        value,
        advantage,
        ret,
        loss_eps,
        loss_t,
        initial_cfm_loss,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.config.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        obs_cache = self.policy.build_obs_cache(obs)

        cfm_loss = compute_cfm_loss(
            self.policy,
            obs,
            action,
            loss_eps=loss_eps,
            loss_t=loss_t,
            obs_cache=obs_cache,
        )
        if self.config.average_losses_before_exp:
            rho_s = torch.exp(initial_cfm_loss.mean(dim=-1) - cfm_loss.mean(dim=-1))
        else:
            rho_s = torch.exp(
                torch.clamp(
                    initial_cfm_loss - cfm_loss,
                    -3.0,
                    3.0,
                )
            ).mean(dim=-1)
        surrogate_loss1 = rho_s * advantage.reshape(-1)
        surrogate_loss2 = (
            torch.clamp(
                rho_s,
                1 - self.config.clipping_epsilon,
                1 + self.config.clipping_epsilon,
            )
            * advantage.reshape(-1)
        )
        policy_loss = -torch.minimum(surrogate_loss1, surrogate_loss2).mean()

        new_value = self.policy._get_value(obs, obs_cache).view(-1)
        value_loss = ((ret.reshape(-1) - new_value) ** 2).mean() * self.config.value_loss_coeff
        total_loss = policy_loss + value_loss
        clipped_ratio_mean = (
            (rho_s.sub(1.0).abs() > self.config.clipping_epsilon)
            .float()
            .mean()
        )
        metrics = dict(
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            total_loss=float(total_loss.detach().cpu()),
            policy_ratio_mean=float(rho_s.mean().detach().cpu()),
            clipped_ratio_mean=float(clipped_ratio_mean.detach().cpu()),
            advantages_mean=float(advantage.mean().detach().cpu()),
            advantages_std=float(advantage.std().detach().cpu()),
        )
        return total_loss, metrics

    def learn_impl(self) -> tuple[int, dict]:
        learn_progress_total = self.get_learn_progress_total()
        learn_progress_current = 0
        metric_history = dict(
            policy_loss=[],
            value_loss=[],
            total_loss=[],
            policy_ratio_mean=[],
            clipped_ratio_mean=[],
            advantages_mean=[],
            advantages_std=[],
        )
        batch_to_device_total = 0.0
        compute_loss_total = 0.0
        backward_step_total = 0.0
        dataloader_total = 0.0
        learn_started_at = time.perf_counter()
        dataloader_started_at = time.perf_counter()
        batch_cache = self._build_train_batch_cache()
        dataloader_total += time.perf_counter() - dataloader_started_at
        obs_all, action_all, value_all, advantage_all, ret_all, loss_eps_all, loss_t_all, initial_cfm_loss_all = batch_cache
        num_items = action_all.shape[0]
        if num_items == 0:
            raise RuntimeError("FPO learn_impl received an empty rollout buffer.")

        for _ in range(self.config.num_updates_per_batch):
            dataloader_started_at = time.perf_counter()
            indices = torch.randperm(num_items, device=self.policy.device)
            dataloader_total += time.perf_counter() - dataloader_started_at
            for i in range(0, num_items, self.config.batch_size):
                batch_indices = indices[i : i + self.config.batch_size]

                batch_to_device_started_at = time.perf_counter()
                batch = (
                    torch_tree_get_item(obs_all, batch_indices),
                    action_all[batch_indices],
                    value_all[batch_indices],
                    advantage_all[batch_indices],
                    ret_all[batch_indices],
                    loss_eps_all[batch_indices],
                    loss_t_all[batch_indices],
                    initial_cfm_loss_all[batch_indices],
                )
                batch_to_device_total += time.perf_counter() - batch_to_device_started_at

                compute_loss_started_at = time.perf_counter()
                total_loss, batch_metrics = self._compute_loss(*batch)
                _sync_cuda_if_needed(self.policy.device)
                compute_loss_total += time.perf_counter() - compute_loss_started_at

                backward_step_started_at = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()
                _sync_cuda_if_needed(self.policy.device)
                backward_step_total += time.perf_counter() - backward_step_started_at
                learn_progress_current += 1
                self.report_learn_progress(learn_progress_current, learn_progress_total)
                for key, value in batch_metrics.items():
                    metric_history[key].append(value)

        if not metric_history["policy_loss"]:
            raise RuntimeError("FPO learn_impl produced no minibatch metrics.")

        self.curr_train_itrs += 1
        self.rollout_buffer.reset()
        learn_loop_time = time.perf_counter() - learn_started_at
        return self.global_step, dict(
            train=dict(train_itrs=float(self.curr_train_itrs)),
            losses=dict(
                policy_loss=float(np.mean(metric_history["policy_loss"])),
                value_loss=float(np.mean(metric_history["value_loss"])),
                total_loss=float(np.mean(metric_history["total_loss"])),
            ),
            fpo=dict(
                policy_ratio_mean=float(np.mean(metric_history["policy_ratio_mean"])),
                clipped_ratio_mean=float(np.mean(metric_history["clipped_ratio_mean"])),
                advantages_mean=float(np.mean(metric_history["advantages_mean"])),
                advantages_std=float(np.mean(metric_history["advantages_std"])),
            ),
            learn_runtime=dict(
                dataloader_time=dataloader_total,
                batch_to_device_time=batch_to_device_total,
                compute_loss_time=compute_loss_total,
                backward_step_time=backward_step_total,
                learn_loop_time=learn_loop_time,
                dataloader_ratio=0.0 if learn_loop_time <= 0 else dataloader_total / learn_loop_time,
                batch_to_device_ratio=0.0
                if learn_loop_time <= 0
                else batch_to_device_total / learn_loop_time,
                compute_loss_ratio=0.0
                if learn_loop_time <= 0
                else compute_loss_total / learn_loop_time,
                backward_step_ratio=0.0
                if learn_loop_time <= 0
                else backward_step_total / learn_loop_time,
            ),
        )
