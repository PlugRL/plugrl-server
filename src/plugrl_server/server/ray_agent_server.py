import asyncio
from collections import deque
from typing import Any
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
import ray

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
)

# Deferred path:
# This Ray server path is maintained only for minimal compatibility.
# Real distributed redesign/debugging is postponed until a true multi-rank environment is available.

from plugrl_server.algorithm.distributed import DDPAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.common.data_utils import batch_aggregate
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import MetricSink
from plugrl_server.common.progress import ProgressReporter
from plugrl_server.policy.state import slice_policy_step_state
from plugrl_server.server.inference_coordinator import InferenceCoordinator
from plugrl_server.server.lifecycle import ServerLifecycle
from plugrl_server.server.protocol import (
    ActionMessage,
    MetadataMessage,
    ProtocolValidationError,
    parse_feedback_request,
    parse_infer_request,
)
from plugrl_server.server.runtime_metrics import RuntimeMetricTracker
from plugrl_server.server.runtime_scheduler import RuntimeScheduler
from plugrl_server.server.training_backend import RayTrainingBackend

logger = get_logger(__name__)


# The scheduler wakes on this interval to look for queued inference requests,
# so it also sets the floor on how long a request waits before anyone sees it.
# At 1 ms that wait was two thirds of the measured round trip. Lowering it to
# 0.1 ms measured 1.76x throughput (457 -> 803 exchanges/s, medians of five
# runs, ranges 425-470 and 692-892), 15% less CPU per exchange, and no change
# in idle CPU. 0.01 ms showed no reliable further gain.
# See experiments/e5-boundary-cost/FINDINGS.md.
SCHEDULER_SLEEP_INTERVAL = 0.0001  # seconds


class ServerStoppingError(RuntimeError):
    pass


class RayAgentServer:
    def __init__(
        self,
        inference_algorithm: DDPAlgorithm,
        checkpoint_manager: CheckpointManager,
        metric_sink: MetricSink,
        learner_actor_ref: ray.ObjectRef,
        show_metric_table: bool = True,
        show_progress_bar: bool = True,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm: DDPAlgorithm = inference_algorithm
        self._checkpoint_manager: CheckpointManager = checkpoint_manager
        self._metric_sink = metric_sink
        self._learner_actor: Any = learner_actor_ref

        self._host = host
        self._port = port
        self._metadata = metadata or {}

        self._inference = InferenceCoordinator(
            stopping_error_factory=ServerStoppingError,
        )
        self._model_lock = asyncio.Lock()
        self._server: Any = None
        self._lifecycle = ServerLifecycle()
        self._scheduler = RuntimeScheduler(
            stop_event=self._lifecycle.stop_event,
            sleep_interval=SCHEDULER_SLEEP_INTERVAL,
        )
        self._progress_reporter = ProgressReporter(enabled=show_progress_bar)
        self._training = RayTrainingBackend(
            algorithm=self._algorithm,
            checkpoint_manager=self._checkpoint_manager,
            metric_sink=self._metric_sink,
            learner_actor_ref=self._learner_actor,
            runtime_metrics_provider=self._runtime_metrics,
            show_metric_table=show_metric_table,
            progress_reporter=self._progress_reporter,
            stop_requested=self._lifecycle.stop_event.is_set,
        )

        self._total_connections = 0
        self._collect_progress_started = False
        self._runtime_metric_tracker = RuntimeMetricTracker()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _request_shutdown(self, reason: str, close_reason: str | None = None) -> None:
        self._lifecycle.request_shutdown(reason, close_reason)

    def _install_signal_handlers(self) -> None:
        self._lifecycle.install_signal_handlers()

    async def _handle_fatal(self, context: str, exc: BaseException) -> None:
        await self._lifecycle.handle_fatal(
            context,
            exc,
            extra_cleanup=self._training.shutdown,
        )

    async def _abort_pending_infer_requests(self, reason: str) -> None:
        (
            pending_futures,
            drained_requests,
        ) = await self._inference.abort_pending_requests(reason)

        if pending_futures > 0 or drained_requests > 0:
            logger.info(
                "Shutdown cleanup finished: "
                f"pending_futures={pending_futures}, drained_requests={drained_requests}"
            )

    async def _shutdown(self, scheduler_task: asyncio.Task, reason: str) -> None:
        self._lifecycle.shutdown_reason = reason
        await self._lifecycle.shutdown(
            scheduler_task=scheduler_task,
            server=self._server,
            abort_pending_infer_requests=self._abort_pending_infer_requests,
            extra_cleanup=self._training.shutdown,
        )
        self._server = None
        self._progress_reporter.close()

    async def run(self):
        scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._install_signal_handlers()
        try:
            self._server = await _server.serve(
                self._handler, self._host, self._port, compression=None, max_size=None
            )
            logger.info(f"Agent Server is listening on {self._host}:{self._port}")
            await self._lifecycle.stop_event.wait()
        except asyncio.CancelledError:
            self._lifecycle.shutdown_reason = "Server run task was cancelled."
            logger.info(
                "Server cancellation received. Closing connections and allowing workers to reconnect."
            )
            raise
        except Exception as exc:
            await self._handle_fatal("agent server runtime", exc)
            raise
        finally:
            await asyncio.shield(
                self._shutdown(scheduler_task, self._lifecycle.shutdown_reason)
            )

    async def _close_for_protocol_error(
        self, websocket: _server.ServerConnection, exc: Exception
    ) -> None:
        logger.warning(f"{exc}. Requesting worker resync.")
        await websocket.close(
            code=websockets.frames.CloseCode.GOING_AWAY,
            reason=SERVER_RESYNC_REASON,
        )

    async def _handler(self, websocket: _server.ServerConnection):
        packer = msgpack_numpy.Packer()
        session_id = str(websocket.remote_address)
        connection_counted = False

        try:
            await websocket.send(
                packer.pack(MetadataMessage(data=self._metadata).to_payload())
            )
            self._total_connections += 1
            connection_counted = True
            prev_node: tuple = (-1, "")
            terminated, truncated = False, False
            action_buffer = deque()
            while True:
                packed_infer_msg = await websocket.recv()
                infer_payload = msgpack_numpy.unpackb(packed_infer_msg)
                try:
                    infer_msg = parse_infer_request(infer_payload)
                except (KeyError, ProtocolValidationError) as exc:
                    await self._close_for_protocol_error(websocket, exc)
                    break

                obs, step_state = infer_msg.data, None

                if not action_buffer:
                    req_id = f"{session_id}-{uuid.uuid4()}"
                    response_future = await self._inference.register_request(req_id)

                    infer_request = dict(id=req_id, obs=obs)
                    await self._inference.enqueue_request(infer_request)

                    try:
                        action, step_state = await response_future
                    except ServerStoppingError:
                        await self._inference.pop_request(req_id)
                        if self._lifecycle.close_reason:
                            try:
                                await websocket.close(
                                    code=websockets.frames.CloseCode.GOING_AWAY,
                                    reason=self._lifecycle.close_reason,
                                )
                            except Exception:
                                pass
                        logger.info(
                            "Shutdown interrupted an in-flight inference request "
                            f"from {websocket.remote_address}."
                        )
                        break
                    else:
                        await self._inference.pop_request(req_id)

                    action_buffer.extend(action.swapaxes(1, 0))
                runtime_state = (
                    step_state.runtime_state if step_state is not None else None
                )
                train_state = step_state.train_state if step_state is not None else None

                if self._algorithm.break_action_chunk:
                    action = action_buffer.popleft()
                else:
                    action = np.array(
                        [action_buffer.popleft() for _ in range(len(action_buffer))]
                    )

                action_response = ActionMessage(data=dict(action=action))
                await websocket.send(packer.pack(action_response.to_payload()))

                packed_feedback_msg = await websocket.recv()
                feedback_payload = msgpack_numpy.unpackb(packed_feedback_msg)
                try:
                    feedback_msg = parse_feedback_request(feedback_payload)
                except (KeyError, ProtocolValidationError) as exc:
                    await self._close_for_protocol_error(websocket, exc)
                    break

                next_obs = feedback_msg.data.obs
                reward = feedback_msg.data.rewards
                next_terminated = feedback_msg.data.terminated
                next_truncated = feedback_msg.data.truncated
                info = feedback_msg.data.info

                feedback_started_at = asyncio.get_running_loop().time()
                async with self._model_lock:
                    self._ensure_collect_progress_started()
                    prev_node, step, log_dict = self._algorithm.feedback(
                        obs=obs,
                        runtime_state=runtime_state,
                        train_state=train_state,
                        terminated=terminated,
                        truncated=truncated,
                        next_obs=next_obs,
                        reward=reward,
                        next_terminated=next_terminated,
                        next_truncated=next_truncated,
                        info=info,
                        prev_node=prev_node,
                    )
                    await asyncio.to_thread(self._training.log, log_dict, step=step)
                self._progress_reporter.update_phase(
                    "collect",
                    completed=self._algorithm.get_collect_progress_completed(),
                    advance=0,
                )
                self._runtime_metric_tracker.observe_feedback(
                    duration=asyncio.get_running_loop().time() - feedback_started_at,
                    batch_size=1,
                )

                terminated, truncated = next_terminated, next_truncated

        except websockets.ConnectionClosed:
            pass
        except Exception as exc:
            try:
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
            except Exception:
                pass
            await self._handle_fatal("connection handler", exc)
        finally:
            if connection_counted:
                self._total_connections = max(0, self._total_connections - 1)

    def _runtime_metrics(self) -> dict:
        return dict(
            server=dict(total_connections=self._total_connections),
            **self._runtime_metric_tracker.as_metrics(),
        )

    def _start_collect_progress(self) -> None:
        self._collect_progress_started = True
        self._progress_reporter.start_phase(
            "collect",
            total=self._algorithm.get_collect_progress_total(),
            description="collect",
        )

    def _ensure_collect_progress_started(self) -> None:
        if (
            not self._collect_progress_started
            and not self._training.should_stop()
            and not self._lifecycle.stop_event.is_set()
        ):
            self._start_collect_progress()

    def should_infer(self) -> bool:
        return (
            self._inference.queue.qsize() >= self._total_connections // 2
            and self._total_connections > 0
        )

    async def _process_infer(self):
        batch = await self._inference.drain_batch()
        if not batch:
            return
        started_at = asyncio.get_running_loop().time()
        try:
            obs = batch_aggregate([req["obs"] for req in batch])
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, runtime_state = self._algorithm.infer(obs)
                step_state = self._algorithm.build_step_state_from_runtime_state(
                    runtime_state,
                    include_train_state=True,
                )
            logger.debug(f"Inference done for batch size {len(batch)}")
            for i, req in enumerate(batch):
                self._inference.resolve_request(
                    req,
                    result=(
                        action[i : i + 1],
                        slice_policy_step_state(step_state, slice(i, i + 1)),
                    ),
                )
            self._runtime_metric_tracker.observe_infer(
                duration=asyncio.get_running_loop().time() - started_at,
                batch_size=len(batch),
            )
        except Exception as exc:
            self._inference.fail_batch(batch, exc)
            raise

    async def _run_control_cycle(self) -> None:
        async with self._model_lock:
            if self._training.should_learn():
                if self._collect_progress_started:
                    self._progress_reporter.finish_phase("collect")
                    self._collect_progress_started = False
                await self._training.process_learn()

            stop_requested = self._training.should_stop()
            if self._training.should_save() or stop_requested:
                await self._training.process_save()

            if stop_requested:
                self._request_shutdown(
                    "Stopping server as the algorithm signaled to stop.",
                    close_reason=SERVER_STOP_REASON,
                )

    async def _scheduler_loop(self):
        try:
            await self._scheduler.run(
                should_infer=self.should_infer,
                process_infer=self._process_infer,
                run_control_cycle=self._run_control_cycle,
            )
        except Exception as exc:
            await self._handle_fatal("scheduler loop", exc)
            raise
