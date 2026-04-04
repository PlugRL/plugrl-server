import asyncio
from typing import Any
from collections.abc import Callable

import ray

# Deferred path:
# Ray-based distributed training support is maintained only for minimal compatibility.
# Real distributed redesign/debugging is postponed until a true multi-rank environment is available.

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm, LearnInterrupted
from plugrl_server.algorithm.distributed import DDPAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager, Checkpoint
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import (
    MetricDict,
    MetricSink,
    merge_metric_groups,
    render_metrics_table,
)
from plugrl_server.common.progress import ProgressTracker
from plugrl_server.common.progress import ProgressReporter

logger = get_logger(__name__)


class LocalTrainingBackend:
    def __init__(
        self,
        *,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        metric_sink: MetricSink,
        runtime_metrics_provider: Callable[[], MetricDict] | None = None,
        show_metric_table: bool = True,
        progress_reporter: ProgressReporter | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._metric_sink = metric_sink
        self._runtime_metrics_provider = runtime_metrics_provider or (lambda: dict())
        self._show_metric_table = show_metric_table
        self._progress_reporter = progress_reporter
        self._stop_requested = stop_requested or (lambda: False)
        self._progress_tracker = ProgressTracker(
            global_steps=algorithm.get_total_training_steps()
        )

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    async def process_learn(self) -> None:
        learn_started_at = self._progress_tracker.mark_learn_start()
        learn_total = self._algorithm.get_learn_progress_total()
        if self._progress_reporter is not None:
            self._progress_reporter.start_phase(
                "learn", total=learn_total, description="learn"
            )
        self._algorithm.set_learn_progress_callback(self._on_learn_progress)
        self._algorithm.set_stop_requested_callback(self._stop_requested)
        try:
            self._algorithm.pre_learn()
            try:
                step, log_dict = await asyncio.to_thread(self._algorithm.learn)
            except LearnInterrupted:
                logger.info("Learn interrupted by shutdown request.")
                return
            self._algorithm.post_learn()
        finally:
            self._algorithm.set_learn_progress_callback(None)
            self._algorithm.set_stop_requested_callback(None)
            if self._progress_reporter is not None:
                self._progress_reporter.finish_phase("learn")
        self.log(
            log_dict,
            step=step,
            progress_snapshot=self._progress_tracker.snapshot_after_learn(
                global_step=step, learn_started_at=learn_started_at
            ),
        )

    async def process_save(self) -> None:
        checkpoint = self._algorithm.create_checkpoint()
        await asyncio.to_thread(self._checkpoint_manager.save_checkpoint, checkpoint)

    def log(self, log_dict: dict, *, step: int, progress_snapshot) -> None:
        merged_metrics = merge_metric_groups(
            log_dict,
            self._runtime_metrics_provider(),
            progress_snapshot.as_metrics(),
        )
        self._metric_sink.log_scalars(merged_metrics, step=step)
        if self._show_metric_table:
            render_metrics_table(
                merged_metrics,
                console=(
                    self._progress_reporter.get_console()
                    if self._progress_reporter is not None
                    else None
                ),
            )

    def _on_learn_progress(self, current: int, total: int | None) -> None:
        if self._progress_reporter is None:
            return
        self._progress_reporter.update_phase(
            "learn", completed=current, total=total, advance=0
        )


class RayTrainingBackend:
    def __init__(
        self,
        *,
        algorithm: DDPAlgorithm,
        checkpoint_manager: CheckpointManager,
        metric_sink: MetricSink,
        learner_actor_ref: ray.ObjectRef,
        runtime_metrics_provider: Callable[[], MetricDict] | None = None,
        show_metric_table: bool = True,
        progress_reporter: ProgressReporter | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._metric_sink = metric_sink
        self._learner_actor: Any = learner_actor_ref
        self._runtime_metrics_provider = runtime_metrics_provider or (lambda: dict())
        self._show_metric_table = show_metric_table
        self._progress_reporter = progress_reporter
        self._stop_requested = stop_requested or (lambda: False)
        self._progress_tracker = ProgressTracker(
            global_steps=algorithm.get_total_training_steps()
        )

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    async def process_learn(self) -> None:
        logger.info("Scheduler initiating distributed training via LearnerActor.")
        learn_started_at = self._progress_tracker.mark_learn_start()
        learn_total = self._algorithm.get_learn_progress_total()
        if self._progress_reporter is not None:
            self._progress_reporter.start_phase(
                "learn", total=learn_total, description="learn"
            )
        self._algorithm.set_stop_requested_callback(self._stop_requested)
        self._algorithm.pre_learn()
        global_step, meta_info, serializable_buffer_data = (
            self._algorithm.get_server_data()
        )
        buffer_data_ref = await asyncio.to_thread(ray.put, serializable_buffer_data)

        learn_ref = self._learner_actor.learn.remote(
            global_step, meta_info, buffer_data_ref
        )
        checkpoint, global_step, train_info = await asyncio.to_thread(
            ray.get, learn_ref
        )

        await self._update_inference_policy(checkpoint)
        if self._progress_reporter is not None:
            self._progress_reporter.finish_phase("learn")
        self._algorithm.set_stop_requested_callback(None)
        await asyncio.to_thread(
            self.log,
            train_info,
            step=global_step,
            progress_snapshot=self._progress_tracker.snapshot_after_learn(
                global_step=global_step, learn_started_at=learn_started_at
            ),
        )
        logger.info(f"Logged training info for step {global_step}.")
        self._algorithm.post_learn()

    async def process_save(self) -> None:
        checkpoint = await asyncio.to_thread(self._algorithm.create_ddp_checkpoint)
        await asyncio.to_thread(self._checkpoint_manager.save_checkpoint, checkpoint)
        logger.info(f"Checkpoint saved locally at step {checkpoint.step}.")

    async def shutdown(self) -> None:
        if ray.is_initialized():
            await asyncio.to_thread(ray.shutdown)

    async def _update_inference_policy(self, checkpoint: Checkpoint) -> None:
        self._algorithm.load_learner_state(checkpoint)
        logger.info("Local inference policy successfully updated.")

    def log(self, log_dict: dict, *, step: int, progress_snapshot) -> None:
        merged_metrics = merge_metric_groups(
            log_dict,
            self._runtime_metrics_provider(),
            progress_snapshot.as_metrics(),
        )
        self._metric_sink.log_scalars(merged_metrics, step=step)
        if self._show_metric_table:
            render_metrics_table(
                merged_metrics,
                console=(
                    self._progress_reporter.get_console()
                    if self._progress_reporter is not None
                    else None
                ),
            )
