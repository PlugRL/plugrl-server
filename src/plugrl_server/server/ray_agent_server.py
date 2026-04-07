import asyncio
import os
import time
from typing import Any
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames
import uuid

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
)

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import MetricSink
from plugrl_server.common.progress import ProgressReporter
from plugrl_server.server.ray_inference import RayInferenceWorkerGroup
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
from plugrl_server.server.training_backend import (
    LocalTrainingBackend,
    RolloutOnlyTrainingBackend,
)

logger = get_logger(__name__)

SCHEDULER_SLEEP_INTERVAL = 0.001  # seconds
INFER_READY_TIMEOUT = 5.0  # seconds to wait for full infer batch before warning
FEEDBACK_WAIT_TIMEOUT = 60.0  # seconds to wait for client feedback before closing
WS_PING_INTERVAL = float(os.environ.get("PLUGRL_WS_PING_INTERVAL_SECONDS", "60"))
WS_PING_TIMEOUT = float(os.environ.get("PLUGRL_WS_PING_TIMEOUT_SECONDS", "180"))
WS_CLOSE_TIMEOUT = float(os.environ.get("PLUGRL_WS_CLOSE_TIMEOUT_SECONDS", "30"))


class ServerStoppingError(RuntimeError):
    pass


class RayAgentServer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        metric_sink: MetricSink,
        inference_workers: RayInferenceWorkerGroup,
        mini_infer_batch_size: int | None = None,
        show_metric_table: bool = True,
        show_progress_bar: bool = True,
        rollout_only: bool = False,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._metric_sink = metric_sink
        self._inference_workers = inference_workers
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
        self._training = (
            RolloutOnlyTrainingBackend()
            if rollout_only
            else LocalTrainingBackend(
                algorithm=self._algorithm,
                checkpoint_manager=self._checkpoint_manager,
                metric_sink=self._metric_sink,
                runtime_metrics_provider=self._runtime_metrics,
                show_metric_table=show_metric_table,
                progress_reporter=self._progress_reporter,
                stop_requested=self._lifecycle.stop_event.is_set,
            )
        )

        self._total_connections = 0
        self._infer_wait_start: float | None = None
        self._collect_progress_started = False
        self._runtime_metric_tracker = RuntimeMetricTracker()
        self._mini_infer_batch_size = mini_infer_batch_size
        self._logged_route_keys: set[str] = set()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _request_shutdown(self, reason: str, close_reason: str | None = None) -> None:
        self._lifecycle.request_shutdown(reason, close_reason)

    def _install_signal_handlers(self) -> None:
        self._lifecycle.install_signal_handlers()

    async def _handle_fatal(self, context: str, exc: BaseException) -> None:
        await self._lifecycle.handle_fatal(context, exc)

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
        )
        self._server = None
        self._progress_reporter.close()

    async def run(self):
        scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._install_signal_handlers()
        try:
            self._server = await _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                close_timeout=WS_CLOSE_TIMEOUT,
            )
            logger.info(
                "Ray Agent Server is listening on %s:%s with %s inference worker(s), ping_interval=%ss, ping_timeout=%ss",
                self._host,
                self._port,
                self._inference_workers.num_workers,
                WS_PING_INTERVAL,
                WS_PING_TIMEOUT,
            )
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
            prev_node_map: dict = {}
            step_state_map: dict = {}
            terminated_map: dict = {}
            truncated_map: dict = {}
            last_obs_map: dict = {}
            while True:
                packed_infer_msg = await websocket.recv()
                infer_payload = msgpack_numpy.unpackb(packed_infer_msg)
                try:
                    infer_msg = parse_infer_request(infer_payload)
                except (KeyError, ProtocolValidationError) as exc:
                    await self._close_for_protocol_error(websocket, exc)
                    break

                obs, env_ids = infer_msg.data, infer_msg.env_indices

                req_id = f"{session_id}-{uuid.uuid4()}"
                response_future = await self._inference.register_request(req_id)
                await self._inference.enqueue_request(
                    dict(
                        id=req_id,
                        obs=obs,
                        env_ids=np.asarray(env_ids),
                        route_key=session_id,
                    )
                )
                try:
                    response = await response_future
                    action_arr, step_state_arr, resp_env_ids, resp_obs = response
                    assert np.array_equal(
                        np.asarray(resp_env_ids), np.asarray(env_ids)
                    ), "Response env_ids do not match request env_ids"
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
                    if session_id not in self._logged_route_keys:
                        self._logged_route_keys.add(session_id)
                        logger.info(
                            "Assigned new inference route session=%s env_ids=%s assignment=%s assignments=%s",
                            session_id,
                            np.asarray(env_ids).tolist(),
                            self._inference_workers.describe_route_assignment(
                                session_id
                            ),
                            self._inference_workers.describe_route_assignments(),
                        )

                resp_obs_list = unbatch_aggregate(resp_obs, aggregate_method="concat")
                for i, env_id in enumerate(resp_env_ids):
                    step_state_map[env_id] = step_state_arr[i]
                    last_obs_map[env_id] = resp_obs_list[i]

                action_response = ActionMessage(
                    data=dict(env_ids=resp_env_ids, action=action_arr.swapaxes(0, 1))
                )
                await websocket.send(packer.pack(action_response.to_payload()))

                try:
                    packed_feedback_msg = await asyncio.wait_for(
                        websocket.recv(), timeout=FEEDBACK_WAIT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out waiting for feedback from %s after %.1fs, closing connection",
                        websocket.remote_address,
                        FEEDBACK_WAIT_TIMEOUT,
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.GOING_AWAY,
                        reason="Feedback timeout",
                    )
                    break

                feedback_payload = msgpack_numpy.unpackb(packed_feedback_msg)
                try:
                    feedback_msg = parse_feedback_request(feedback_payload)
                except (KeyError, ProtocolValidationError) as exc:
                    await self._close_for_protocol_error(websocket, exc)
                    break

                fb_env_ids = feedback_msg.env_indices
                next_obs_batch = feedback_msg.data.obs
                reward_list = feedback_msg.data.rewards
                next_terminated_list = feedback_msg.data.terminated
                next_truncated_list = feedback_msg.data.truncated
                info_batch = feedback_msg.data.info
                next_obs_list = unbatch_aggregate(
                    next_obs_batch, aggregate_method="concat"
                )
                info_list = unbatch_aggregate(info_batch, aggregate_method="stack")
                assert len(info_list) == len(next_obs_list) or len(info_list) == 0

                feedback_started_at = time.perf_counter()
                async with self._model_lock:
                    self._ensure_collect_progress_started()
                    for idx, env_id in enumerate(fb_env_ids):
                        next_obs = next_obs_list[idx]
                        reward = reward_list[idx]
                        next_terminated = next_terminated_list[idx]
                        next_truncated = next_truncated_list[idx]
                        info = info_list[idx] if len(info_list) > 0 else {}

                        prev_node = prev_node_map.get(env_id, (-1, ""))
                        step_state = step_state_map.get(env_id)
                        runtime_state = (
                            step_state.runtime_state if step_state is not None else None
                        )
                        train_state = (
                            step_state.train_state if step_state is not None else None
                        )
                        terminated = terminated_map.get(env_id, False)
                        truncated = truncated_map.get(env_id, False)
                        last_obs = last_obs_map.get(env_id, {})

                        prev_node_res, step, log_dict = self._algorithm.feedback(
                            obs=last_obs,
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

                        prev_node_map[env_id] = prev_node_res
                        terminated_map[env_id] = bool(next_terminated)
                        truncated_map[env_id] = bool(next_truncated)
                        self._metric_sink.log_scalars(log_dict, step=step)
                    self._progress_reporter.update_phase(
                        "collect",
                        completed=self._algorithm.get_collect_progress_completed(),
                        advance=0,
                    )
                self._runtime_metric_tracker.observe_feedback(
                    duration=time.perf_counter() - feedback_started_at,
                    batch_size=len(fb_env_ids),
                )

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
            **self._inference_workers.as_metrics(),
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
        current_qsize = self._inference.queue.qsize()
        current_env_count = self._inference.queued_env_count()
        now = time.monotonic()

        infer_threshold = self._mini_infer_batch_size or self._total_connections
        ready_count = (
            current_env_count if self._mini_infer_batch_size else current_qsize
        )
        if (ready_count >= infer_threshold) and self._total_connections > 0:
            self._infer_wait_start = None
            return True

        if current_qsize > 0 and self._total_connections > 0:
            if self._infer_wait_start is None:
                self._infer_wait_start = now
            elif now - self._infer_wait_start > INFER_READY_TIMEOUT:
                logger.warning(
                    "Infer queue has been waiting %.1fs for %s/%s ready units (requests=%s, envs=%s)",
                    now - self._infer_wait_start,
                    ready_count,
                    infer_threshold,
                    current_qsize,
                    current_env_count,
                )
                self._infer_wait_start = now
        else:
            self._infer_wait_start = None

        return False

    async def _process_infer(self):
        batch = await self._inference.drain_batch()
        if not batch:
            return
        started_at = time.perf_counter()
        try:
            async with self._model_lock:
                responses = await self._inference_workers.infer_batch(batch)
            for request, response in zip(batch, responses, strict=True):
                logger.info(
                    "Ray inference served request_id=%s envs=%s worker=%s model_step=%s",
                    request["id"],
                    len(request["env_ids"]),
                    response["worker_index"],
                    response["model_step"],
                )
                self._inference.resolve_request(
                    request,
                    result=(
                        response["action"],
                        response["step_states"],
                        response["env_ids"],
                        response["obs"],
                    ),
                )
            self._runtime_metric_tracker.observe_infer(
                duration=time.perf_counter() - started_at,
                batch_size=sum(len(request["env_ids"]) for request in batch),
            )
        except Exception as exc:
            self._inference.fail_batch(batch, exc)
            raise

    async def _sync_inference_workers(self) -> None:
        checkpoint = self._algorithm.create_checkpoint()
        await self._inference_workers.sync_checkpoint(checkpoint)
        await self._inference_workers.refresh_metrics()

    async def _run_control_cycle(self) -> None:
        learned = False
        async with self._model_lock:
            if self._training.should_learn():
                if self._collect_progress_started:
                    self._progress_reporter.finish_phase("collect")
                    self._collect_progress_started = False
                await self._training.process_learn()
                learned = True

        if learned:
            await self._sync_inference_workers()

        async with self._model_lock:
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
            await self._handle_fatal("main scheduler loop", exc)
            raise
