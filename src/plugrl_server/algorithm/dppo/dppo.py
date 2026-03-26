import dataclasses
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import tqdm

from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.data_utils import torch_tree_to_device
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.algorithm.base_algorithm import BaseAlgorithm, BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo, register_algo_config
from plugrl_server.policy.state import PolicyRuntimeState, PolicyTrainState, to_numpy_state
from plugrl_server.policy.base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicy,
    DiffusionRuntimeState,
)

from .dppo_buffer import DPPOBuffer
from plugrl_server.buffer.rollout_buffer import ROLLOUT_BUFFER_SCHEMA_VERSION

logger = get_logger(__name__)

try:
    import dppo.util.scheduler as _dppo_scheduler
except ImportError:
    raise ImportError(
        'dppo is not installed. Please install it with pip install "plugrl-server[dppo]".'
    )

UID = "dppo"


@dataclasses.dataclass
class SchedulerConfig:
    min_lr: float
    warmup_steps: int = 0


@register_algo_config(UID)
@dataclasses.dataclass
class DPPOAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.99
    """total timesteps of the experiments"""
    gamma_denoising: float = 1.0
    """the discount factor for denoising"""

    actor_lr: float = 1e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 1e-3
    """the learning rate of the critic optimizer"""
    actor_weight_decay: float = 0.0
    """the weight decay of the actor optimizer"""
    critic_weight_decay: float = 0.0
    """the weight decay of the critic optimizer"""
    actor_lr_scheduler: SchedulerConfig | None = None
    """the learning rate scheduler of the actor optimizer"""
    critic_lr_scheduler: SchedulerConfig | None = None
    """the learning rate scheduler of the critic optimizer"""

    buffer_size: int = 20000
    """the total size of the buffer"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    update_epochs: int = 4
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
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float | None = None
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = 1
    """the target KL divergence threshold"""

    logprob_noise_level: float = 0.01
    sampling_noise_level: float = 0.01
    clip_advantage_lower_quantile: float = 0
    clip_advantage_upper_quantile: float = 1
    n_critic_warmup_itrs: int = 0
    use_normalized_rewards: bool = False

    batch_size: int = 256
    critic_batch_size: int | None = None
    train_itrs: int = 200
    save_interval: int = 10
    """number of steps to accumulate gradients over before calling optimizer.step()"""
    grad_accum_steps: int = 8

    @property
    def total_steps(self) -> int:
        return self.buffer_size * self.train_itrs

    def __post_init__(self):
        if self.critic_batch_size is None:
            self.critic_batch_size = self.batch_size


@register_algo(UID)
class DPPOAlgorithm(BaseAlgorithm):
    config: DPPOAlgoConfig
    policy: BasePolicyGradientDiffusionPolicy
    break_action_chunk: bool = False
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer | None

    def __init__(
        self, config: DPPOAlgoConfig, policy: BasePolicyGradientDiffusionPolicy
    ):
        super().__init__(config, policy)
        logger.info("Initializing DPPO Buffer...")
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
        logger.info("DPPOAlgorithm initialized")
        self.save_interval = config.save_interval
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0
        self._episode_stats = deque()
        self._buffer_pbar = tqdm.tqdm(
            total=self.rollout_buffer.buffer_size,
            ncols=0,
            leave=False,
            smoothing=0.01,
        )

    @property
    def active_policy(self) -> BasePolicyGradientDiffusionPolicy:
        return self.policy

    def init_optimizers(self):
        logger.info("Initializing DPPO Optimizers and Schedulers...")
        config = self.config
        self.actor_optimizer = torch.optim.AdamW(
            self.active_policy.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        if config.actor_lr_scheduler is not None:
            self.actor_lr_scheduler = _dppo_scheduler.CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=config.train_itrs,
                max_lr=config.actor_lr,
                min_lr=config.actor_lr_scheduler.min_lr,
                warmup_steps=config.actor_lr_scheduler.warmup_steps,
            )
        else:
            self.actor_lr_scheduler = None
        if self.active_policy.critic is not None:
            self.critic_optimizer = torch.optim.AdamW(
                self.active_policy.critic.parameters(),
                lr=config.critic_lr,
                weight_decay=config.critic_weight_decay,
            )
            if config.critic_lr_scheduler is not None:
                self.critic_lr_scheduler = (
                    _dppo_scheduler.CosineAnnealingWarmupRestarts(
                        self.critic_optimizer,
                        first_cycle_steps=config.train_itrs,
                        max_lr=config.critic_lr,
                        min_lr=config.critic_lr_scheduler.min_lr,
                        warmup_steps=config.critic_lr_scheduler.warmup_steps,
                    )
                )
            else:
                self.critic_lr_scheduler = None
        else:
            self.critic_optimizer = None
            self.critic_lr_scheduler = None

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            action, runtime_state = self.active_policy.get_action_and_runtime_state(
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

    def example_train_state(self, batch_size: int) -> PolicyTrainState:
        return self.derive_train_state(self.active_policy.fake_runtime_state(batch_size))

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
        if train_state is None:
            assert runtime_state is not None, (
                "Runtime state must be provided when train_state is missing."
            )
            assert isinstance(runtime_state, DiffusionRuntimeState), (
                "DPPO expects diffusion runtime state."
            )
            assert runtime_state.obs is not None
            train_state = self.derive_train_state(runtime_state)
        assert train_state is not None, "DPPO requires train_state for rollout storage."
        current_node = self.rollout_buffer.add_frame(
            prev_node=prev_node,
            train_state=train_state,
            reward=reward,
            done=truncated or terminated,
            last_value=None,
            next_done=next_truncated or next_terminated,
        )
        self.rollout_buffer.add_next_obs_value_request(
            obs=next_obs, end_node=current_node
        )
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info:
                self._record_episode_stats(info["episode"])
                self._buffer_pbar.set_description(
                    self._format_buffer_desc(self._current_episode_metrics())
                )
            self.rollout_buffer.finish_rollout(info=info)
        self.global_step += 1
        self._buffer_pbar.update(1)
        return current_node, self.global_step, log_dict

    def pre_learn(self) -> None:
        self.rollout_buffer.compute_advantages_and_returns(
            policy=self.active_policy, batch_size=self.config.batch_size
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

    def _optimizer_step_if_ready(
        self, accum_steps: int, grad_accum: int, force: bool = False
    ) -> tuple[float | None, float | None]:
        if force or accum_steps % grad_accum == 0:
            max_actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.active_policy.actor.parameters(), float("inf")
            ).item()
            max_critic_grad_norm = 0.0
            if self.active_policy.critic is not None:
                max_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.active_policy.critic.parameters(), float("inf")
                ).item()

            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.active_policy.parameters(), self.config.max_grad_norm
                )
            if self.curr_train_itrs >= self.config.n_critic_warmup_itrs:
                self.actor_optimizer.step()
            if self.critic_optimizer is not None:
                self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad()

            return max_actor_grad_norm, max_critic_grad_norm
        return None, None

    def _compute_loss(self, obs, action, oldlogprob, reward, value, advantage, ret):
        """Compute policy loss, value loss, and entropy for a batch."""
        batch_size, ft_denoising_steps = action.shape[:2]
        x, t, cond = (
            obs["x"].reshape(-1, *obs["x"].shape[2:]),
            obs["t"].reshape(-1),
            obs["cond"],
        )

        # Policy forward step
        _, newlogprob, entropy = self.active_policy._denoising_step(
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

        if self.active_policy.critic is not None:
            newvalue = self.active_policy._get_value(obs["cond"]).view(-1)
            if self.config.clip_vloss_coef is not None:
                v_loss_unclipped = (newvalue - ret) ** 2
                v_clipped = value + torch.clamp(
                    newvalue - value,
                    -self.config.clip_vloss_coef,
                    self.config.clip_vloss_coef,
                )
                v_loss_clipped = (v_clipped - ret) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - ret) ** 2).mean()
        else:
            v_loss = torch.tensor(0.0)

        return pg_loss, v_loss, entropy_loss, logratio, ratio

    def learn(self) -> tuple[int, dict]:
        sampler, dataloader = self.create_dataloaders()
        description = self.rollout_buffer.description()
        v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = torch.tensor(0.0)
        clipfracs = []
        max_actor_grad_norms, max_critic_grad_norms = [], []

        grad_accum = max(1, int(self.config.grad_accum_steps))
        for update_epoch in range(self.config.update_epochs):
            logger.info(
                f"DPPO Update Epoch {update_epoch + 1}/{self.config.update_epochs}"
            )
            if sampler is not None:
                sampler.set_epoch(update_epoch)  # type: ignore

            break_flag = False
            self.actor_optimizer.zero_grad()
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad()
            accum_steps = 0

            for batch in dataloader:
                obs, action, oldlogprob, reward, value, advantage, ret = batch
                obs = torch_tree_to_device(obs, self.active_policy.device)
                action = action.to(self.active_policy.device)
                oldlogprob = oldlogprob.to(self.active_policy.device)
                reward = reward.to(self.active_policy.device)
                value = value.to(self.active_policy.device)
                advantage = advantage.to(self.active_policy.device)
                ret = ret.to(self.active_policy.device)
                pg_loss, v_loss, entropy_loss, logratio, ratio = self._compute_loss(
                    obs, action, oldlogprob, reward, value, advantage, ret
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
                loss = loss / grad_accum
                loss.backward()
                accum_steps += 1

                max_actor_grad_norm, max_critic_grad_norm = (
                    self._optimizer_step_if_ready(accum_steps, grad_accum)
                )
                if max_actor_grad_norm is not None:
                    max_actor_grad_norms.append(max_actor_grad_norm)
                    max_critic_grad_norms.append(max_critic_grad_norm)

                if (
                    self.config.target_kl is not None
                    and approx_kl > self.config.target_kl
                ):
                    break_flag = True
                    logger.info(
                        f"Early stopping at epoch {update_epoch} due to reaching max KL."
                    )
                    break

            # flush remaining gradients
            if accum_steps % grad_accum != 0:
                max_actor_grad_norm, max_critic_grad_norm = (
                    self._optimizer_step_if_ready(accum_steps, grad_accum, force=True)
                )
                if max_actor_grad_norm is not None:
                    max_actor_grad_norms.append(max_actor_grad_norm)
                    max_critic_grad_norms.append(max_critic_grad_norm)

            if break_flag:
                break

        # Step schedulers
        if (
            self.actor_lr_scheduler is not None
            and self.curr_train_itrs >= self.config.n_critic_warmup_itrs
        ):
            self.actor_lr_scheduler.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()

        # Logging
        train_info = {
            "charts/actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "charts/critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"]
            if self.critic_optimizer is not None
            else 0.0,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "train/global_step": self.global_step,
            "train/actor_max_grad_norm": max(max_actor_grad_norms)
            if max_actor_grad_norms
            else 0.0,
            "train/critic_max_grad_norm": max(max_critic_grad_norms)
            if max_critic_grad_norms
            else 0.0,
            "train/train_itrs": self.curr_train_itrs,
        }
        train_info.update(description)
        train_info.update(self._current_episode_metrics())
        self.curr_train_itrs += 1

        return self.global_step, train_info

    def post_learn(self) -> None:
        self.rollout_buffer.reset()
        self._episode_stats.clear()
        self._buffer_pbar.reset(total=self.rollout_buffer.buffer_size)
        self._buffer_pbar.set_description(
            self._format_buffer_desc(self._current_episode_metrics())
        )

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
                "critic": self.critic_optimizer.state_dict()
                if self.critic_optimizer is not None
                else None,
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
            self.active_policy.load_state_dict(checkpoint.model)
        if checkpoint.optimizer is not None:
            self.actor_optimizer.load_state_dict(checkpoint.optimizer["actor"])
            if (
                self.critic_optimizer is not None
                and checkpoint.optimizer["critic"] is not None
            ):
                self.critic_optimizer.load_state_dict(checkpoint.optimizer["critic"])
        if "train_itrs" in checkpoint.meta:
            self.curr_train_itrs = checkpoint.meta["train_itrs"]
        if "last_saved_itr" in checkpoint.meta:
            self.last_saved_itr = checkpoint.meta["last_saved_itr"]
        logger.info(
            f"Loaded checkpoint at step {self.global_step}, train_itrs {self.curr_train_itrs}, last_saved_itr {self.last_saved_itr}"
        )

    def _record_episode_stats(self, episode_info: dict) -> None:
        success = float(episode_info.get("s", 0.0))
        reward = float(episode_info.get("r", 0.0))
        length = float(episode_info.get("l", 0.0))
        self._episode_stats.append((success, reward, length))

    def _current_episode_metrics(self) -> dict[str, float]:
        if not self._episode_stats:
            return {"train/success": 0.0, "train/reward": 0.0, "train/length": 0.0}
        stats = np.asarray(self._episode_stats, dtype=np.float32)
        return {
            "train/success": stats[:, 0].mean(),
            "train/reward": stats[:, 1].mean(),
            "train/length": stats[:, 2].mean(),
        }

    def _format_buffer_desc(self, metrics: dict[str, float]) -> str:
        return ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
