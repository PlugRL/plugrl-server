import math
import time

import numpy as np
import torch

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.algorithm.master_weights import MasterWeights
from plugrl_server.algorithm.registration import register_algo
from plugrl_server.common.checkpoint_manager import (
    Checkpoint,
    load_checkpoint_from_path,
)
from plugrl_server.common.logging_utils import get_logger
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
from plugrl_server.policy.state import (
    PolicyRuntimeState,
    PolicyTrainState,
    to_numpy_state,
)

from .fpo_buffer import FPOBuffer
from .fpo_config import FPOAlgoConfig, UID
from .utils import compute_cfm_loss

logger = get_logger(__name__)


# How many recent minibatches the drift stop averages over. Small enough to
# react within an iteration, large enough that one noisy minibatch does not
# end it.
_DRIFT_WINDOW = 8


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@register_algo(UID)
class FPOAlgorithm(BaseAlgorithm):
    config: FPOAlgoConfig
    policy: BasePolicyGradientFlowPolicy

    # The spread of the advantages as they come out of the buffer, before any
    # normalising. NaN until the first learn step has built a batch cache.
    _advantage_raw_std: float = float("nan")

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
            treat_truncated_as_done=config.treat_truncated_as_done,
        )
        # Pi0Policy holds its action expert in bfloat16, where a step at a
        # small learning rate rounds away. The optimizer steps float32 copies
        # of such parameters instead; see MasterWeights.
        self.master_weights = MasterWeights(
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters()),
            device=config.master_weights_device,
        )
        self.optimizer = torch.optim.Adam(
            self.master_weights.optimizer_params,
            lr=config.learning_rate,
        )
        self.global_step = 0
        self.curr_train_itrs = 0
        self.last_saved_itr = 0
        if config.policy_checkpoint_path is not None:
            self._restore_from(config.policy_checkpoint_path, config.restore)

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
        cond = numpy_state["obs"]["cond"]
        return dict(
            # A vector observation is stored as float32, as it always was. A
            # tree - Pi0Policy's images, masks and token ids - keeps each
            # leaf's dtype: a dict cannot be cast at all, and uint8 images cast
            # to float32 would take four times the buffer.
            obs=(
                np.asarray(cond, dtype=np.float32)
                if isinstance(cond, np.ndarray)
                else cond
            ),
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
        return current_node, self.global_step, dict()

    def pre_learn(self) -> None:
        self.rollout_buffer.prepare_fpo_fields(
            self.policy,
            batch_size=self.config.batch_size,
            output_mode=self.config.output_mode,
        )
        if not self.config.fpo_playground_trick:
            self.rollout_buffer.compute_advantages_and_returns(
                policy=self.policy,
                batch_size=self.config.batch_size,
            )
        if isinstance(self.policy, FPOPolicy):
            obs = self.rollout_buffer.train_state_storage.get_item(slice(None))
            obs_torch = torch_tree_to_device(
                numpy_tree_to_torch(obs), self.policy.device
            )
            self.policy.update_obs_stats(obs_torch)

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

    def post_learn(self) -> None:
        super().post_learn()
        # A learn step allocates far more than inference does: master weights,
        # optimizer state, the buffer's observations, attention intermediates.
        # PyTorch keeps those blocks in its caching allocator when the step
        # ends, and collection is what comes next. E11's second training
        # iteration died looking for 124 MiB for an attention matmul with the
        # card at 24,098 MiB of 24,125 MiB, none of it leaked. Hand the cache
        # back before inference resumes.
        device = getattr(self.policy, "device", None)
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            self.master_weights.sync_from_model()
        if checkpoint.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer)
        self.global_step = checkpoint.step
        self.curr_train_itrs = int(checkpoint.meta.get("curr_train_itrs", 0))
        self.last_saved_itr = self.curr_train_itrs

    def _restore_from(self, path, restore: str) -> None:
        """Start this run from a saved one, taking as much of it as asked.

        `load_checkpoint` takes everything, which is what resuming means. The
        other two modes exist to reproduce a start that is not a resume: a
        policy that is already good arriving in front of an optimizer, or a
        value head, that has never seen it. E14's pi0.5 started that way and
        lost 29 of 50 in a single iteration; nothing on the FPO path could
        reproduce that shape on a small model, because nothing could load
        weights into a training run at all.
        """
        logger.info("Restoring from %s with restore=%s", path, restore)
        checkpoint = load_checkpoint_from_path(path)
        if restore == "all":
            self.load_checkpoint(checkpoint)
            return

        if checkpoint.model is None:
            raise ValueError(f"checkpoint at {path} carries no model weights")
        state = dict(checkpoint.model)
        if restore == "except-critic":
            # Dropped, not zeroed: the keys left out keep whatever the policy
            # built for them, which is the random initialisation a fresh value
            # head has. `obs_stats_*` are not under `critic.` and so stay.
            dropped = [k for k in state if k.startswith("critic.")]
            for key in dropped:
                del state[key]
            logger.info("Holding back %d critic tensors", len(dropped))
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"checkpoint at {path} has unknown keys: {unexpected}")
        if restore == "model" and missing:
            raise ValueError(f"checkpoint at {path} is missing keys: {missing}")
        self.master_weights.sync_from_model()
        # Optimizer, step and iteration count are deliberately left as built.

    def _build_train_batch_cache(self) -> tuple:
        idx = len(self.rollout_buffer)
        obs = self.rollout_buffer.train_state_storage.get_item(slice(None, idx))
        obs_torch = torch_tree_to_device(numpy_tree_to_torch(obs), self.policy.device)
        action = torch.from_numpy(self.rollout_buffer.actions[:idx]).to(
            self.policy.device
        )
        truncated = torch.from_numpy(
            self.rollout_buffer.truncated[:idx].astype(np.float32)
        ).to(self.policy.device)
        loss_eps = torch.from_numpy(self.rollout_buffer.loss_eps[:idx]).to(
            self.policy.device
        )
        loss_t = torch.from_numpy(self.rollout_buffer.loss_t[:idx]).to(
            self.policy.device
        )
        initial_cfm_loss = torch.from_numpy(
            self.rollout_buffer.initial_cfm_loss[:idx]
        ).to(self.policy.device)
        _sync_cuda_if_needed(self.policy.device)
        return (
            obs_torch,
            action,
            truncated,
            loss_eps,
            loss_t,
            initial_cfm_loss,
        )

    def _refresh_epoch_value_targets(
        self, obs_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = len(self.rollout_buffer)
        value_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, idx, self.config.batch_size):
                j = min(i + self.config.batch_size, idx)
                obs_batch = torch_tree_get_item(obs_all, slice(i, j))
                obs_cache_batch = self.policy.build_obs_cache(obs_batch)
                value_batch = self.policy._get_value(obs_batch, obs_cache_batch)
                # Pi0Policy's value head runs in bfloat16, which numpy cannot hold.
                value_batches.append(
                    value_batch.detach().float().cpu().numpy().reshape(-1, 1)
                )
        self.rollout_buffer.values[:idx] = np.concatenate(value_batches, axis=0)
        self.rollout_buffer.compute_advantages_and_returns(
            policy=self.policy,
            batch_size=self.config.batch_size,
        )
        value = torch.from_numpy(self.rollout_buffer.values[:idx]).to(
            self.policy.device
        )
        advantage = torch.from_numpy(self.rollout_buffer.advantages[:idx]).to(
            self.policy.device
        )
        ret = torch.from_numpy(self.rollout_buffer.returns[:idx]).to(self.policy.device)
        return value, self._scale_advantage(advantage), ret

    def _scale_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        """Normalise over the whole buffer, not per minibatch.

        This used to happen inside `_compute_loss`, which receives a minibatch.
        That is the usual arrangement and it is fine at the batch sizes this
        algorithm was written for - its own default is 1024. A pi0.5-sized
        policy does not fit those. E14 ran at batch 8, forced by memory, and at
        8 the arrangement stops being a rescaling and becomes a source of
        signal: the standard deviation of eight samples estimates nothing, and
        subtracting their mean forces half of every eight positive and half
        negative at unit scale whatever the rewards were.

        What this does NOT fix, stated so the change is not credited with more
        than it earns: normalising rescales whatever spread it finds to 1, so a
        buffer holding only the value head's own variation still comes out at
        unit scale. Once a policy has collapsed far enough that no episode
        earns a reward, that is all there is. Guarding that needs a rule for
        when a buffer carries no signal, which is a separate change. This one
        addresses the cause, not the floor a collapse settles onto.
        """
        # Recorded before the division, because afterwards the spread is 1.0 by
        # construction and says nothing. This is the statistic that shows
        # whether the buffer held a signal: E14 had no reward anywhere from its
        # third iteration on and none of its metrics could show it, because the
        # only advantage statistic logged was measured after normalising.
        self._advantage_raw_std = float(advantage.std().detach().cpu())
        if not self.config.normalize_advantage:
            return advantage
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    def in_critic_warmup(self) -> bool:
        """Is this iteration one where only the value head should learn?

        `curr_train_itrs` counts iterations already finished, so it is 0
        throughout the first one.
        """
        return self.curr_train_itrs < self.config.n_critic_warmup_itrs

    def _zero_actor_grads(self) -> None:
        """Drop the actor's gradients between backward and the optimizer.

        Not `requires_grad = False` and not dropping the policy loss from the
        total: the value loss reaches the actor as well, through the
        observation cache its own backbone builds, so a policy-loss-free total
        would still move the actor. Zeroing after backward is the one point
        where every path into the actor has already been taken and none of it
        has been applied yet.

        Adam carries no momentum past this either. A parameter whose gradient
        is zero on every step of the warmup has zero first and second moments,
        and the optimizer runs without weight decay, so the actor comes out of
        the warmup exactly as it went in.
        """
        for param in self.policy.actor.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def _policy_has_drifted(self, metric_history: dict[str, list[float]]) -> bool:
        """Has the policy moved too far from the one that collected the data?

        Clipping bounds what one sample contributes to one update. It does not
        bound where a few thousand updates end up, and PPO implementations
        pair it with a stop for exactly that. Measured here: the mean policy
        ratio stays at 0.99 and the clipped fraction falls while the policy
        walks away from a solution it had already found.

        The recent window rather than the whole history, because the early
        minibatches of an iteration sit at a ratio of 1 by construction and
        would mask a late excursion.
        """
        limit = self.config.max_policy_drift
        if limit <= 0:
            return False
        recent = metric_history["policy_ratio_mean"][-_DRIFT_WINDOW:]
        if len(recent) < _DRIFT_WINDOW:
            return False
        return abs(1.0 - float(np.mean(recent))) > limit

    def _compute_loss(
        self,
        obs,
        action,
        truncated,
        value,
        advantage,
        ret,
        loss_eps,
        loss_t,
        initial_cfm_loss,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
        # `advantage` arrives already normalised over the whole buffer, in
        # `_build_train_batch_cache`. It is deliberately not normalised again
        # here: this method sees one minibatch, and at the batch sizes a VLA
        # forces, a per-minibatch statistic manufactures signal rather than
        # removing scale.
        obs_cache_started_at = time.perf_counter()
        obs_cache = self.policy.build_obs_cache(obs)
        _sync_cuda_if_needed(self.policy.device)
        obs_cache_time = time.perf_counter() - obs_cache_started_at

        cfm_loss_started_at = time.perf_counter()
        cfm_loss = compute_cfm_loss(
            self.policy,
            obs,
            action,
            output_mode=self.config.output_mode,
            loss_eps=loss_eps,
            loss_t=loss_t,
            obs_cache=obs_cache,
        )
        _sync_cuda_if_needed(self.policy.device)
        cfm_loss_time = time.perf_counter() - cfm_loss_started_at
        loss_delta = (initial_cfm_loss - cfm_loss).clamp(-3, 3)

        ratio_policy_loss_started_at = time.perf_counter()
        if self.config.average_losses_before_exp:
            rho_s = torch.exp(loss_delta.mean(dim=-1))
        else:
            rho_s = torch.exp(
                torch.clamp(
                    loss_delta,
                    -3.0,
                    3.0,
                )
            ).mean(dim=-1)
        surrogate_loss1 = rho_s * advantage.reshape(-1)
        surrogate_loss2 = torch.clamp(
            rho_s,
            1 - self.config.clipping_epsilon,
            1 + self.config.clipping_epsilon,
        ) * advantage.reshape(-1)
        policy_loss = -torch.minimum(surrogate_loss1, surrogate_loss2).mean()
        _sync_cuda_if_needed(self.policy.device)
        ratio_policy_loss_time = time.perf_counter() - ratio_policy_loss_started_at

        value_forward_started_at = time.perf_counter()
        new_value = self.policy._get_value(obs, obs_cache).view(-1)
        v_error = ret.reshape(-1) - new_value
        if self.config.fpo_playground_trick:
            v_error = v_error * (1.0 - truncated.reshape(-1))
        value_loss = (v_error**2).mean() * self.config.value_loss_coeff
        total_loss = policy_loss + value_loss
        _sync_cuda_if_needed(self.policy.device)
        value_forward_time = time.perf_counter() - value_forward_started_at
        clipped_ratio_mean = (
            (rho_s.sub(1.0).abs() > self.config.clipping_epsilon).float().mean()
        )
        metrics = dict(
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            total_loss=float(total_loss.detach().cpu()),
            policy_ratio_mean=float(rho_s.mean().detach().cpu()),
            clipped_ratio_mean=float(clipped_ratio_mean.detach().cpu()),
            advantages_mean=float(advantage.mean().detach().cpu()),
            advantages_std=float(advantage.std().detach().cpu()),
            initial_cfm_loss_mean=float(initial_cfm_loss.mean().detach().cpu()),
            cfm_loss_mean=float(cfm_loss.mean().detach().cpu()),
            loss_delta_mean=float(loss_delta.mean().detach().cpu()),
            loss_delta_min=float(loss_delta.min().detach().cpu()),
            loss_delta_max=float(loss_delta.max().detach().cpu()),
        )
        loss_runtime = dict(
            obs_cache_time=obs_cache_time,
            cfm_loss_time=cfm_loss_time,
            ratio_policy_loss_time=ratio_policy_loss_time,
            value_forward_time=value_forward_time,
        )
        return total_loss, metrics, loss_runtime

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
            initial_cfm_loss_mean=[],
            cfm_loss_mean=[],
            loss_delta_mean=[],
            loss_delta_min=[],
            loss_delta_max=[],
        )
        batch_to_device_total = 0.0
        compute_loss_total = 0.0
        stopped_early = False
        # Read once, here: `curr_train_itrs` is incremented at the end of this
        # method, so asking again where the metrics are assembled would report
        # the next iteration's answer.
        in_warmup = self.in_critic_warmup()
        backward_step_total = 0.0
        dataloader_total = 0.0
        obs_cache_total = 0.0
        cfm_loss_total = 0.0
        ratio_policy_loss_total = 0.0
        value_forward_total = 0.0
        learn_started_at = time.perf_counter()
        dataloader_started_at = time.perf_counter()
        batch_cache = self._build_train_batch_cache()
        dataloader_total += time.perf_counter() - dataloader_started_at
        (
            obs_all,
            action_all,
            truncated_all,
            loss_eps_all,
            loss_t_all,
            initial_cfm_loss_all,
        ) = batch_cache
        num_items = action_all.shape[0]
        if num_items == 0:
            raise RuntimeError("FPO learn_impl received an empty rollout buffer.")

        if not self.config.fpo_playground_trick:
            value_all = torch.from_numpy(self.rollout_buffer.values[:num_items]).to(
                self.policy.device
            )
            advantage_all = self._scale_advantage(
                torch.from_numpy(self.rollout_buffer.advantages[:num_items]).to(
                    self.policy.device
                )
            )
            ret_all = torch.from_numpy(self.rollout_buffer.returns[:num_items]).to(
                self.policy.device
            )

        for _ in range(self.config.num_updates_per_batch):
            if self.config.fpo_playground_trick:
                dataloader_started_at = time.perf_counter()
                value_all, advantage_all, ret_all = self._refresh_epoch_value_targets(
                    obs_all
                )
                dataloader_total += time.perf_counter() - dataloader_started_at
            dataloader_started_at = time.perf_counter()
            indices = torch.randperm(num_items, device=self.policy.device)
            dataloader_total += time.perf_counter() - dataloader_started_at
            for i in range(0, num_items, self.config.batch_size):
                batch_indices = indices[i : i + self.config.batch_size]

                batch_to_device_started_at = time.perf_counter()
                batch = (
                    torch_tree_get_item(obs_all, batch_indices),
                    action_all[batch_indices],
                    truncated_all[batch_indices],
                    value_all[batch_indices],
                    advantage_all[batch_indices],
                    ret_all[batch_indices],
                    loss_eps_all[batch_indices],
                    loss_t_all[batch_indices],
                    initial_cfm_loss_all[batch_indices],
                )
                batch_to_device_total += (
                    time.perf_counter() - batch_to_device_started_at
                )

                compute_loss_started_at = time.perf_counter()
                total_loss, batch_metrics, loss_runtime = self._compute_loss(*batch)
                _sync_cuda_if_needed(self.policy.device)
                compute_loss_total += time.perf_counter() - compute_loss_started_at
                obs_cache_total += loss_runtime["obs_cache_time"]
                cfm_loss_total += loss_runtime["cfm_loss_time"]
                ratio_policy_loss_total += loss_runtime["ratio_policy_loss_time"]
                value_forward_total += loss_runtime["value_forward_time"]

                backward_step_started_at = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                self.master_weights.clear_model_grads()
                total_loss.backward()
                if self.in_critic_warmup():
                    self._zero_actor_grads()
                self.master_weights.grads_to_masters()
                self.optimizer.step()
                self.master_weights.masters_to_model()
                _sync_cuda_if_needed(self.policy.device)
                backward_step_total += time.perf_counter() - backward_step_started_at
                learn_progress_current += 1
                self.report_learn_progress(learn_progress_current, learn_progress_total)
                for key, value in batch_metrics.items():
                    metric_history[key].append(value)

                if self._policy_has_drifted(metric_history):
                    stopped_early = True
                    break
            if stopped_early:
                break

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
                advantages_raw_std=self._advantage_raw_std,
                # 1 while only the value head was learning, so a reader of the
                # curves can see which iterations moved the policy at all.
                critic_warmup=float(in_warmup),
                initial_cfm_loss_mean=float(
                    np.mean(metric_history["initial_cfm_loss_mean"])
                ),
                cfm_loss_mean=float(np.mean(metric_history["cfm_loss_mean"])),
                loss_delta_mean=float(np.mean(metric_history["loss_delta_mean"])),
                loss_delta_min=float(np.min(metric_history["loss_delta_min"])),
                loss_delta_max=float(np.max(metric_history["loss_delta_max"])),
            ),
            learn_runtime=dict(
                dataloader_time=dataloader_total,
                batch_to_device_time=batch_to_device_total,
                compute_loss_time=compute_loss_total,
                backward_step_time=backward_step_total,
                learn_loop_time=learn_loop_time,
                obs_cache_time=obs_cache_total,
                cfm_loss_time=cfm_loss_total,
                ratio_policy_loss_time=ratio_policy_loss_total,
                value_forward_time=value_forward_total,
                dataloader_ratio=0.0
                if learn_loop_time <= 0
                else dataloader_total / learn_loop_time,
                batch_to_device_ratio=0.0
                if learn_loop_time <= 0
                else batch_to_device_total / learn_loop_time,
                compute_loss_ratio=0.0
                if learn_loop_time <= 0
                else compute_loss_total / learn_loop_time,
                backward_step_ratio=0.0
                if learn_loop_time <= 0
                else backward_step_total / learn_loop_time,
                obs_cache_ratio=0.0
                if learn_loop_time <= 0
                else obs_cache_total / learn_loop_time,
                cfm_loss_ratio=0.0
                if learn_loop_time <= 0
                else cfm_loss_total / learn_loop_time,
                ratio_policy_loss_ratio=0.0
                if learn_loop_time <= 0
                else ratio_policy_loss_total / learn_loop_time,
                value_forward_ratio=0.0
                if learn_loop_time <= 0
                else value_forward_total / learn_loop_time,
            ),
        )
