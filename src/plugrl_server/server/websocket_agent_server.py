import asyncio
import time
from typing import Any
import numpy as np
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
from plugrl_server.common.data_utils import batch_aggregate, unbatch_aggregate
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import MetricSink
from plugrl_server.common.progress import ProgressReporter
from plugrl_server.policy.state import slice_policy_step_state
from plugrl_server.server.inference_coordinator import InferenceCoordinator
from plugrl_server.server.lifecycle import ServerLifecycle
from plugrl_server.server.metadata import build_server_metadata
from plugrl_server.server.protocol import (
    ActionMessage,
    MetadataMessage,
    ProtocolValidationError,
    parse_feedback_request,
    parse_infer_request,
)
from plugrl_server.server.runtime_metrics import RuntimeMetricTracker
from plugrl_server.server.runtime_scheduler import RuntimeScheduler
from plugrl_server.server.training_backend import LocalTrainingBackend

logger = get_logger(__name__)

# The scheduler wakes on this interval to look for queued inference requests,
# so it also sets the floor on how long a request waits before anyone sees it.
# At 1 ms that wait was two thirds of the measured round trip. Lowering it to
# 0.1 ms measured 1.76x throughput (457 -> 803 exchanges/s, medians of five
# runs, ranges 425-470 and 692-892), 15% less CPU per exchange, and no change
# in idle CPU. 0.01 ms showed no reliable further gain.
# See experiments/e5-boundary-cost/FINDINGS.md.
SCHEDULER_SLEEP_INTERVAL = 0.0001  # seconds
INFER_READY_TIMEOUT = 5.0  # seconds to wait for full infer batch before warning
FEEDBACK_WAIT_TIMEOUT = 60.0  # seconds to wait for client feedback before closing
# How long a shutdown will wait for the model lock in order to write a final
# checkpoint. Long enough for a learn step to notice it has been asked to
# stop, short enough that Ctrl-C still feels like Ctrl-C.
SHUTDOWN_SAVE_TIMEOUT = 30.0  # seconds


class ServerStoppingError(RuntimeError):
    pass


class WebSocketAgentServer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        metric_sink: MetricSink,
        mini_infer_batch_size: int | None = None,
        show_metric_table: bool = True,
        show_progress_bar: bool = True,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._metric_sink = metric_sink
        self._host = host
        self._port = port
        # SPEC.md section 5.1: the client reads this before it can send
        # anything, so it is the only place it can learn the action shape
        # without being told out of band. Anything the caller passes wins.
        self._metadata = build_server_metadata(algorithm, extra=metadata)

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
        self._training = LocalTrainingBackend(
            algorithm=self._algorithm,
            checkpoint_manager=self._checkpoint_manager,
            metric_sink=self._metric_sink,
            runtime_metrics_provider=self._runtime_metrics,
            show_metric_table=show_metric_table,
            progress_reporter=self._progress_reporter,
            stop_requested=self._lifecycle.stop_event.is_set,
        )

        self._total_connections = 0
        self._infer_wait_start: float | None = None
        self._collect_progress_started = False
        self._runtime_metric_tracker = RuntimeMetricTracker()

        self._mini_infer_batch_size = mini_infer_batch_size

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

    async def _save_on_exit(self) -> None:
        """Write a checkpoint on the way out, so an interrupted run resumes.

        Periodic saving is the algorithm's decision and the gap can be very
        large: FPO's default is one save per ten learn cycles, and at its
        default buffer size that is 9.8 million environment steps - hours on
        a CPU. A run stopped before the first of those lost everything, and
        stopping a run early is the normal case, not the exception.

        Three things this deliberately does not do. It does not save after a
        fatal error, because the state that produced one is not state worth
        resuming from. It does not save when the algorithm stopped of its own
        accord, because `_run_control_cycle` has just written a checkpoint at
        that same step and doing it again only writes the model twice and
        logs it twice. And it does not wait indefinitely: the model lock may
        be held by a learn step, which is asked to stop but may take a moment
        to notice, and a shutdown that hangs is worse than a lost checkpoint.
        """
        if self._lifecycle.fatal_reported:
            return
        if self._lifecycle.close_reason == SERVER_STOP_REASON:
            return
        try:
            await asyncio.wait_for(
                self._save_under_lock(), timeout=SHUTDOWN_SAVE_TIMEOUT
            )
            logger.info("Checkpoint written during shutdown.")
        except asyncio.TimeoutError:
            logger.warning(
                f"Gave up writing a shutdown checkpoint after "
                f"{SHUTDOWN_SAVE_TIMEOUT:.0f}s; the model lock was still held."
            )
        except Exception as exc:  # noqa: BLE001 - never block shutdown
            logger.warning(f"Could not write a checkpoint during shutdown: {exc}")

    async def _save_under_lock(self) -> None:
        async with self._model_lock:
            await self._training.process_save()

    async def _shutdown(self, scheduler_task: asyncio.Task, reason: str) -> None:
        self._lifecycle.shutdown_reason = reason
        await self._save_on_exit()
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

                infer_request = dict(
                    id=req_id,
                    obs=obs,
                    env_ids=np.asarray(env_ids),
                )
                await self._inference.enqueue_request(infer_request)

                try:
                    # response now may include env_ids and obs_list for per-env mapping
                    # response is expected to be a tuple:
                    # (action_arr, step_state_arr, resp_env_ids, resp_obs_list)
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
                # register per-env step states and last observations
                resp_obs_list = unbatch_aggregate(resp_obs, aggregate_method="concat")
                for i, eid in enumerate(resp_env_ids):
                    step_state_map[eid] = step_state_arr[i]
                    last_obs_map[eid] = resp_obs_list[i]

                action_response = ActionMessage(
                    data=dict(env_ids=resp_env_ids, action=action_arr.swapaxes(0, 1)),
                )
                await websocket.send(packer.pack(action_response.to_payload()))

                try:
                    packed_feedback_msg = await asyncio.wait_for(
                        websocket.recv(), timeout=FEEDBACK_WAIT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timed out waiting for feedback from {websocket.remote_address} after {FEEDBACK_WAIT_TIMEOUT:.1f}s, closing connection"
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
                # process each env's feedback individually
                async with self._model_lock:
                    self._ensure_collect_progress_started()
                    for idx, eid in enumerate(fb_env_ids):
                        n_obs = next_obs_list[idx]
                        rew = reward_list[idx]
                        n_term = next_terminated_list[idx]
                        n_trunc = next_truncated_list[idx]
                        inf = info_list[idx] if len(info_list) > 0 else {}

                        prev_node = prev_node_map.get(eid, (-1, ""))
                        step_state = step_state_map.get(eid, None)
                        runtime_state = (
                            step_state.runtime_state if step_state is not None else None
                        )
                        train_state = (
                            step_state.train_state if step_state is not None else None
                        )
                        terminated = terminated_map.get(eid, False)
                        truncated = truncated_map.get(eid, False)
                        last_obs = last_obs_map.get(eid, {})
                        prev_node_res, step, log_dict = self._algorithm.feedback(
                            obs=last_obs,
                            runtime_state=runtime_state,
                            train_state=train_state,
                            terminated=terminated,
                            truncated=truncated,
                            next_obs=n_obs,
                            reward=rew,
                            next_terminated=n_term,
                            next_truncated=n_trunc,
                            info=inf,
                            prev_node=prev_node,
                        )

                        prev_node_map[eid] = prev_node_res
                        terminated_map[eid] = bool(n_term)
                        truncated_map[eid] = bool(n_trunc)

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

        # allow queue size to be >= total connections to support per-connection
        # multi-env requests that expand into multiple queue entries

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
                    f"Infer queue has been waiting {now - self._infer_wait_start:.1f}s "
                    f"for {ready_count}/{infer_threshold} ready units "
                    f"(requests={current_qsize}, envs={current_env_count})"
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
            obs = batch_aggregate(
                [req["obs"] for req in batch], aggregate_method="concat"
            )
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, runtime_state = self._algorithm.infer(obs)
                step_state = self._algorithm.build_step_state_from_runtime_state(
                    runtime_state,
                    include_train_state=True,
                )
            logger.debug(f"Inference done for batch size {len(batch)}")
            start = 0
            for req in batch:
                req_size = len(req["env_ids"])
                end = start + req_size
                self._inference.resolve_request(
                    req,
                    result=(
                        action[start:end],
                        [
                            slice_policy_step_state(step_state, slice(i, i + 1))
                            for i in range(start, end)
                        ],
                        req["env_ids"],
                        req["obs"],
                    ),
                )
                start = end
            self._runtime_metric_tracker.observe_infer(
                duration=time.perf_counter() - started_at,
                batch_size=sum(len(req["env_ids"]) for req in batch),
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
