"""PPO, as CleanRL's `ppo_continuous_action.py` runs it.

Clipped surrogate at 0.2, a clipped value loss, advantages normalised per
minibatch, one Adam (eps 1e-5) over the actor and the critic together, the
whole gradient clipped at norm 0.5, the learning rate annealed linearly to
zero. It drives `gaussian-policy`, and any policy that offers the same four
things: `get_action_and_runtime_state`, `evaluate_actions`, `get_value` and,
optionally, `update_obs_stats`.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.algorithm.registration import register_algo
from plugrl_server.algorithm.train_utils import move_batch_to_device
from plugrl_server.buffer.rollout_buffer import ROLLOUT_BUFFER_SCHEMA_VERSION
from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.policy.gaussian.gaussian_policy import GaussianPolicy
from plugrl_server.policy.state import (
    PolicyRuntimeState,
    PolicyTrainState,
    to_numpy_state,
)

from .ppo_buffer import PPOBuffer
from .ppo_config import UID, PPOAlgoConfig

logger = get_logger(__name__)


@register_algo(UID)
class PPOAlgorithm(BaseAlgorithm):
    config: PPOAlgoConfig
    policy: GaussianPolicy
    optimizer: torch.optim.Optimizer

    def __init__(self, config: PPOAlgoConfig, policy: GaussianPolicy):
        super().__init__(config, policy)
        if not hasattr(policy, "evaluate_actions"):
            raise TypeError(
                f"ppo needs a policy with evaluate_actions, such as "
                f"gaussian-policy; {type(policy).__name__} has none"
            )
        if getattr(policy.config, "deterministic", False):
            raise ValueError(
                "ppo needs a policy that samples: --policy.deterministic is for "
                "evaluation, and a mean action has no density to form a ratio from"
            )
        self.rollout_buffer = PPOBuffer(
            buffer_size=config.buffer_size,
            example_train_state=self.example_train_state(batch_size=1),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_rewards=config.normalize_rewards,
            reward_clip=config.reward_clip,
            reward_scaling_gamma=config.reward_scaling_gamma,
        )
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0

    def _critic_param_ids(self) -> set[int]:
        return {id(p) for p in self.policy.critic.parameters()}

    def _actor_params(self) -> list[torch.nn.Parameter]:
        critic = self._critic_param_ids()
        return [p for p in self.policy.parameters() if id(p) not in critic]

    def init_optimizers(self) -> None:
        config = self.config
        if config.critic_learning_rate is None:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.learning_rate, eps=config.adam_eps
            )
            return
        self.optimizer = torch.optim.Adam(
            [
                dict(params=self._actor_params(), lr=config.learning_rate),
                dict(
                    params=list(self.policy.critic.parameters()),
                    lr=config.critic_learning_rate,
                ),
            ],
            eps=config.adam_eps,
        )

    def in_critic_warmup(self) -> bool:
        return self.curr_train_itrs < self.config.n_critic_warmup_itrs

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            return self.policy.get_action_and_runtime_state(obs)

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        numpy_state = to_numpy_state(runtime_state)
        if not isinstance(numpy_state, dict):
            raise TypeError("ppo requires a mapping-like train_state export.")
        return numpy_state

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
        next_terminated: bool,
        next_truncated: bool,
        info: dict,
        prev_node: tuple,
    ) -> tuple[tuple, int, dict]:
        assert train_state is not None, "ppo requires train_state for rollout storage."
        current_node = self.rollout_buffer.add_frame(
            prev_node=prev_node,
            train_state=train_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            last_value=None,
            next_terminated=next_terminated,
            next_truncated=next_truncated,
        )
        self.rollout_buffer.add_next_obs_value_request(
            obs=next_obs, end_node=current_node
        )
        if next_terminated or next_truncated:
            if "episode" in info and bool(info["episode"].get("mask", True)):
                self.record_episode_metrics(info["episode"])
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1
        return current_node, self.global_step, {}

    def pre_learn(self) -> None:
        self.rollout_buffer.compute_advantages_and_returns(
            policy=self.policy, batch_size=self.config.batch_size
        )

    def get_collect_progress_total(self) -> int | None:
        return self.config.buffer_size

    def get_collect_progress_completed(self) -> int | None:
        return len(self.rollout_buffer)

    def get_learn_progress_total(self) -> int | None:
        return self.config.update_epochs * math.ceil(
            self.config.buffer_size / self.config.batch_size
        )

    def _anneal_fraction(self) -> float:
        """CleanRL's: lr = (1 - (iteration - 1) / num_iterations) * learning_rate."""
        if not self.config.anneal_lr:
            return 1.0
        return 1.0 - self.curr_train_itrs / self.config.train_itrs

    def _learning_rate(self) -> float:
        return self._anneal_fraction() * self.config.learning_rate

    def _loss(self, obs, action, oldlogprob, value, advantage, ret):
        config = self.config
        newlogprob, entropy, newvalue = self.policy.evaluate_actions(obs, action)
        logratio = newlogprob - oldlogprob
        ratio = logratio.exp()

        if config.norm_adv and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        pg_loss = torch.max(
            -advantage * ratio,
            -advantage * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef),
        ).mean()

        if config.clip_vloss:
            v_clipped = value + torch.clamp(
                newvalue - value, -config.clip_coef, config.clip_coef
            )
            v_loss = (
                0.5 * torch.max((newvalue - ret) ** 2, (v_clipped - ret) ** 2).mean()
            )
        else:
            v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

        return pg_loss, v_loss, entropy.mean(), logratio, ratio

    def learn_impl(self) -> tuple[int, dict]:
        config = self.config
        lr = self._learning_rate()
        frac = self._anneal_fraction()
        base = [config.learning_rate, config.critic_learning_rate]
        for group, rate in zip(self.optimizer.param_groups, base):
            group["lr"] = frac * rate
        in_warmup = self.in_critic_warmup()
        actor_params = self._actor_params() if in_warmup else []

        dataloader = torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=self.rollout_buffer.collate_fn,
        )
        rollout_summary = self.rollout_buffer.description()
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
        old_approx_kl = approx_kl = torch.tensor(0.0)
        clipfracs: list[float] = []
        grad_norms: list[float] = []
        progress_total = self.get_learn_progress_total()
        progress = 0

        for epoch in range(config.update_epochs):
            for batch in dataloader:
                obs, action, oldlogprob, _reward, value, advantage, ret = (
                    move_batch_to_device(batch, device=self.policy.device)
                )
                pg_loss, v_loss, entropy_loss, logratio, ratio = self._loss(
                    obs, action, oldlogprob, value, advantage, ret
                )
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    )

                loss = (
                    pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                # During a critic warmup the actor's parameters get no
                # gradient, so the optimizer skips them entirely.
                for param in actor_params:
                    param.grad = None
                grad_norms.append(self._clip_or_measure())
                self.optimizer.step()
                progress += 1
                self.report_learn_progress(progress, progress_total)

            if config.target_kl is not None and approx_kl > config.target_kl:
                logger.info("Stopping after epoch %d: approx_kl %.4f", epoch, approx_kl)
                break

        self.curr_train_itrs += 1
        return self.global_step, dict(
            models=dict(learning_rate=lr),
            losses=dict(
                value_loss=v_loss.item(),
                policy_loss=pg_loss.item(),
                entropy=entropy_loss.item(),
                old_approx_kl=old_approx_kl.item(),
                approx_kl=approx_kl.item(),
                clipfrac=float(np.mean(clipfracs)) if clipfracs else 0.0,
            ),
            train=dict(
                max_grad_norm=max(grad_norms) if grad_norms else 0.0,
                train_itrs=float(self.curr_train_itrs),
                critic_warmup=float(in_warmup),
            ),
            rollout=rollout_summary,
        )

    def _clip_or_measure(self) -> float:
        """The gradient's norm, clipped to `max_grad_norm` when that is set."""
        params = [p for p in self.policy.parameters() if p.grad is not None]
        if self.config.max_grad_norm is not None:
            return float(nn.utils.clip_grad_norm_(params, self.config.max_grad_norm))
        norms = torch.stack([p.grad.detach().norm() for p in params])
        return float(torch.linalg.vector_norm(norms))

    def post_learn(self) -> None:
        # After learning and before the reset, as DPPO does for fpo-policy:
        # this iteration collected and learned under one normalisation, and
        # the next collects under the new one.
        update = getattr(self.policy, "update_obs_stats", None)
        filled = len(self.rollout_buffer)
        if update is not None and filled > 0:
            stored = self.rollout_buffer.train_state_storage.get_item(slice(0, filled))
            update(torch.as_tensor(stored, device=self.policy.device))
        self.rollout_buffer.reset()
        super().post_learn()

    def should_learn(self) -> bool:
        return self.rollout_buffer.full()

    def should_stop(self) -> bool:
        return self.curr_train_itrs >= self.config.train_itrs

    def should_save(self) -> bool:
        return (self.curr_train_itrs % self.config.save_interval == 0) and (
            self.curr_train_itrs > self.last_saved_itr
        )

    def create_checkpoint(self) -> Checkpoint:
        self.last_saved_itr = self.curr_train_itrs
        rms = self.rollout_buffer.ret_rms
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer={"adam": self.optimizer.state_dict()},
            meta={
                "train_itrs": self.curr_train_itrs,
                "last_saved_itr": self.last_saved_itr,
                # The reward scale, which a resume would otherwise relearn.
                "return_rms": {
                    "mean": float(rms.mean),
                    "var": float(rms.var),
                    "count": float(rms.count),
                },
                "rollout_buffer_schema_version": ROLLOUT_BUFFER_SCHEMA_VERSION,
            },
        )

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer["adam"])
        meta = checkpoint.meta
        self.curr_train_itrs = meta.get("train_itrs", self.curr_train_itrs)
        self.last_saved_itr = meta.get("last_saved_itr", self.last_saved_itr)
        if "return_rms" in meta:
            rms = self.rollout_buffer.ret_rms
            rms.mean = np.float64(meta["return_rms"]["mean"])
            rms.var = np.float64(meta["return_rms"]["var"])
            rms.count = meta["return_rms"]["count"]
        logger.info(
            "Loaded checkpoint at step %d, train_itrs %d",
            self.global_step,
            self.curr_train_itrs,
        )
