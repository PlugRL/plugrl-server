import numpy as np
import torch
import math

from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.algorithm.registration import register_algo
from plugrl_server.algorithm.train_utils import (
    move_batch_to_device,
    optimizer_step_if_ready,
)
from plugrl_server.policy.state import (
    PolicyRuntimeState,
    PolicyTrainState,
    to_numpy_state,
)
from plugrl_server.policy.base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicy,
    DiffusionRuntimeState,
)

from .dppo_buffer import DPPOBuffer
from .dppo_config import DPPOAlgoConfig, UID
from .dppo_optimizer import build_adamw, build_scheduler
from plugrl_server.buffer.rollout_buffer import ROLLOUT_BUFFER_SCHEMA_VERSION

logger = get_logger(__name__)


@register_algo(UID)
class DPPOAlgorithm(BaseAlgorithm):
    config: DPPOAlgoConfig
    policy: BasePolicyGradientDiffusionPolicy
    break_action_chunk: bool = False
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer

    def __init__(
        self, config: DPPOAlgoConfig, policy: BasePolicyGradientDiffusionPolicy
    ):
        super().__init__(config, policy)
        if policy.critic is None:
            raise ValueError("DPPO requires a critic.")
        example_train_state = self.example_train_state(batch_size=1)
        if example_train_state is None:
            raise ValueError("DPPO requires non-empty train_state for rollout storage.")
        self.rollout_buffer = DPPOBuffer(
            buffer_size=config.buffer_size,
            example_train_state=example_train_state,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            use_normalized_rewards=config.use_normalized_rewards,
        )
        self.save_interval = config.save_interval
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0

    def init_optimizers(self):
        config = self.config
        self.actor_optimizer = build_adamw(
            self.policy.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        self.actor_lr_scheduler = build_scheduler(
            self.actor_optimizer,
            scheduler_config=config.actor_lr_scheduler,
            train_itrs=config.train_itrs,
            max_lr=config.actor_lr,
        )
        self.critic_optimizer = build_adamw(
            self.policy.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.critic_weight_decay,
        )
        self.critic_lr_scheduler = build_scheduler(
            self.critic_optimizer,
            scheduler_config=config.critic_lr_scheduler,
            train_itrs=config.train_itrs,
            max_lr=config.critic_lr,
        )

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            action, runtime_state = self.policy.get_action_and_runtime_state(
                obs, sampling_noise_level=self.config.sampling_noise_level
            )
        return action, runtime_state

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        assert isinstance(runtime_state, DiffusionRuntimeState), (
            "DPPO expects diffusion runtime state."
        )
        numpy_state = to_numpy_state(runtime_state)
        if numpy_state is None or not isinstance(numpy_state, dict):
            raise TypeError("DPPO requires mapping-like train_state export.")
        assert runtime_state.obs is not None
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
        assert train_state is not None, "DPPO requires train_state for rollout storage."
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
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info and bool(info["episode"].get("mask", True)):
                self.record_episode_metrics(info["episode"])
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1
        return current_node, self.global_step, log_dict

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

    def create_dataloaders(
        self,
    ) -> tuple[torch.utils.data.Sampler | None, torch.utils.data.DataLoader]:
        return None, torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=self.rollout_buffer.collate_fn,
        )

    def _compute_loss(self, obs, action, oldlogprob, value, advantage, ret):
        """Compute policy loss, value loss, and entropy for a batch."""
        batch_size, ft_denoising_steps = action.shape[:2]
        x, t, cond = (
            obs["x"].reshape(-1, *obs["x"].shape[2:]),
            obs["t"].reshape(-1),
            obs["cond"],
        )

        # Policy forward step
        _, newlogprob, entropy = self.policy._denoising_step(
            x=x,
            t=t,
            cond=cond,
            x_next=action.reshape(-1, *action.shape[2:]),
            sampling_noise_level=self.config.logprob_noise_level,
        )

        newlogprob = (
            newlogprob.clamp(min=-5, max=2)
            .mean(dim=(-1, -2))
            .reshape(batch_size, ft_denoising_steps)
        )
        oldlogprob = (
            oldlogprob.clamp(min=-5, max=2)
            .mean(dim=(-1, -2))
            .reshape(batch_size, ft_denoising_steps)
        )

        if self.config.norm_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Clip advantage by quantiles
        adv_min = torch.quantile(advantage, self.config.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantage, self.config.clip_advantage_upper_quantile)
        advantage = torch.clamp(advantage, adv_min, adv_max)

        # Apply discount over denoising steps
        denoising_inds = torch.arange(ft_denoising_steps, device=advantage.device)
        discount = self.config.gamma_denoising ** (
            ft_denoising_steps - denoising_inds - 1
        )
        advantage = advantage.unsqueeze(1) * discount.unsqueeze(0)

        # Compute ratio and clip coefficients
        logratio = newlogprob - oldlogprob
        ratio = logratio.exp()
        t_norm = denoising_inds.float() / max(1, ft_denoising_steps - 1)
        clip_ploss_coef = (
            (
                self.config.clip_ploss_coef_base
                + (self.config.clip_ploss_coef - self.config.clip_ploss_coef_base)
                * (torch.exp(self.config.clip_ploss_coef_rate * t_norm) - 1)
                / (np.exp(self.config.clip_ploss_coef_rate) - 1)
            )
            if ft_denoising_steps > 1
            else t_norm
        )
        clip_ploss_coef = clip_ploss_coef.unsqueeze(0)

        # Compute losses
        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * torch.clamp(
            ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()

        newvalue = self.policy._get_value(obs["cond"]).view(-1)
        v_loss_unclipped = (newvalue - ret) ** 2
        v_clipped = value + torch.clamp(
            newvalue - value,
            -self.config.clip_vloss_coef,
            self.config.clip_vloss_coef,
        )
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        return pg_loss, v_loss, entropy_loss, logratio, ratio

    def _step_optimizers(
        self,
        *,
        accum_steps: int,
        grad_accum_steps: int,
        actor_enabled: bool,
        force: bool = False,
        accumulated: int | None = None,
    ) -> tuple[float | None, float | None]:
        max_actor_grad_norm = (
            optimizer_step_if_ready(
                optimizer=self.actor_optimizer,
                parameters=self.policy.actor.parameters(),
                accum_steps=accum_steps,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=self.config.max_grad_norm,
                force=force,
                accumulated=accumulated,
            )
            if actor_enabled
            else None
        )
        if not actor_enabled and force:
            self.actor_optimizer.zero_grad()

        max_critic_grad_norm = optimizer_step_if_ready(
            optimizer=self.critic_optimizer,
            parameters=self.policy.critic.parameters(),
            accum_steps=accum_steps,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=self.config.max_grad_norm,
            force=force,
            accumulated=accumulated,
        )
        return max_actor_grad_norm, max_critic_grad_norm

    def learn_impl(self) -> tuple[int, dict]:
        sampler, dataloader = self.create_dataloaders()
        rollout_summary = self.rollout_buffer.description()
        v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = torch.tensor(0.0)
        clipfracs = []
        max_actor_grad_norms, max_critic_grad_norms = [], []
        actor_enabled = self.curr_train_itrs >= self.config.n_critic_warmup_itrs
        learn_progress_total = self.get_learn_progress_total()
        learn_progress_current = 0

        grad_accum = max(1, int(self.config.grad_accum_steps))
        for update_epoch in range(self.config.update_epochs):
            if sampler is not None:
                sampler.set_epoch(update_epoch)  # type: ignore

            break_flag = False
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            accum_steps = 0
            # Backward passes sitting in the gradients, waiting for a step.
            pending = 0

            for batch in dataloader:
                obs, action, oldlogprob, _reward, value, advantage, ret = (
                    move_batch_to_device(batch, device=self.policy.device)
                )
                pg_loss, v_loss, entropy_loss, logratio, ratio = self._compute_loss(
                    obs, action, oldlogprob, value, advantage, ret
                )

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.config.clip_ploss_coef)
                        .float()
                        .mean()
                        .item()
                    )

                loss = (
                    pg_loss
                    - self.config.ent_coef * entropy_loss
                    + self.config.vf_coef * v_loss
                )
                loss.backward()
                accum_steps += 1
                pending += 1
                learn_progress_current += 1
                self.report_learn_progress(learn_progress_current, learn_progress_total)

                max_actor_grad_norm, max_critic_grad_norm = self._step_optimizers(
                    accum_steps=accum_steps,
                    grad_accum_steps=grad_accum,
                    actor_enabled=actor_enabled,
                    accumulated=pending,
                )
                # The critic always steps when a window closes; the actor does
                # not during a warmup, so it is the critic that says whether
                # the gradients were consumed.
                if max_critic_grad_norm is not None:
                    pending = 0
                if max_actor_grad_norm is not None:
                    max_actor_grad_norms.append(max_actor_grad_norm)
                    max_critic_grad_norms.append(max_critic_grad_norm)

                if approx_kl > self.config.target_kl:
                    break_flag = True
                    logger.info(
                        f"Early stopping at epoch {update_epoch} due to reaching max KL."
                    )
                    break

            # flush remaining gradients
            if pending > 0:
                max_actor_grad_norm, max_critic_grad_norm = self._step_optimizers(
                    accum_steps=accum_steps,
                    grad_accum_steps=grad_accum,
                    actor_enabled=actor_enabled,
                    force=True,
                    accumulated=pending,
                )
                pending = 0
                if max_actor_grad_norm is not None:
                    max_actor_grad_norms.append(max_actor_grad_norm)
                    max_critic_grad_norms.append(max_critic_grad_norm)

            if break_flag:
                break

        # Step schedulers
        if actor_enabled:
            self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        self.curr_train_itrs += 1

        return self.global_step, dict(
            models=dict(
                actor_learning_rate=self.actor_optimizer.param_groups[0]["lr"],
                critic_learning_rate=self.critic_optimizer.param_groups[0]["lr"],
            ),
            losses=dict(
                value_loss=v_loss.item(),
                policy_loss=pg_loss.item(),
                entropy=entropy_loss.item(),
                old_approx_kl=old_approx_kl.item(),
                approx_kl=approx_kl.item(),
                clipfrac=float(np.mean(clipfracs)),
            ),
            train=dict(
                actor_max_grad_norm=max(max_actor_grad_norms)
                if max_actor_grad_norms
                else 0.0,
                critic_max_grad_norm=max(max_critic_grad_norms)
                if max_critic_grad_norms
                else 0.0,
                train_itrs=float(self.curr_train_itrs),
            ),
            rollout=rollout_summary,
        )

    def post_learn(self) -> None:
        self.rollout_buffer.reset()
        super().post_learn()

    def should_learn(self) -> bool:
        return self.rollout_buffer.full()

    def should_stop(self) -> bool:
        return self.curr_train_itrs >= self.config.train_itrs

    def should_save(self) -> bool:
        return (self.curr_train_itrs % self.save_interval == 0) and (
            self.curr_train_itrs > self.last_saved_itr
        )

    def create_checkpoint(self) -> Checkpoint:
        self.last_saved_itr = self.curr_train_itrs
        return Checkpoint(
            step=self.global_step,
            model=self.policy.state_dict(),
            optimizer={
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
            },
            meta={
                "train_itrs": self.curr_train_itrs,
                "last_saved_itr": self.last_saved_itr,
                "rollout_buffer_schema_version": ROLLOUT_BUFFER_SCHEMA_VERSION,
            },
        )

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.global_step = checkpoint.step
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.actor_optimizer.load_state_dict(checkpoint.optimizer["actor"])
            self.critic_optimizer.load_state_dict(checkpoint.optimizer["critic"])
        if "train_itrs" in checkpoint.meta:
            self.curr_train_itrs = checkpoint.meta["train_itrs"]
        if "last_saved_itr" in checkpoint.meta:
            self.last_saved_itr = checkpoint.meta["last_saved_itr"]
        logger.info(
            f"Loaded checkpoint at step {self.global_step}, train_itrs {self.curr_train_itrs}, last_saved_itr {self.last_saved_itr}"
        )
