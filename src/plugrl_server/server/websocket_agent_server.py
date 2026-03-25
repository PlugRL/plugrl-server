import asyncio
import time
from typing import Any
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    MessageType,
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
)

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.common.data_utils import batch_aggregate, unbatch_aggregate
from plugrl_server.policy.state import PolicyStepState, slice_policy_step_state
from plugrl_server.server.inference_coordinator import InferenceCoordinator
from plugrl_server.server.lifecycle import ServerLifecycle
from plugrl_server.server.protocol import (
    ActionMessage,
    MetadataMessage,
    ProtocolValidationError,
    parse_feedback_request,
    parse_infer_request,
)
from plugrl_server.server.runtime_scheduler import RuntimeScheduler
from plugrl_server.server.training_backend import LocalTrainingBackend

SCHEDULER_SLEEP_INTERVAL = 0.001  # seconds
INFER_READY_TIMEOUT = 5.0  # seconds to wait for full infer batch before warning
FEEDBACK_WAIT_TIMEOUT = 60.0  # seconds to wait for client feedback before closing


class ServerStoppingError(RuntimeError):
    pass


class WebSocketAgentServer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        writer: SummaryWriter,
        mini_infer_batch_size: int | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._writer = writer
        self._host = host
        self._port = port
        self._metadata = metadata or {}

        self._inference = InferenceCoordinator(
            stopping_error_factory=ServerStoppingError,
        )
        # Accumulate partial inference outputs per req_id; only fulfill the
        # corresponding future once all sub-indices for that req_id arrive.
        self._pending_infer_results: dict[str, dict[str, Any]] = {}
        self._model_lock = asyncio.Lock()
        self._server: Any = None
        self._lifecycle = ServerLifecycle()
        self._scheduler = RuntimeScheduler(
            stop_event=self._lifecycle.stop_event,
            sleep_interval=SCHEDULER_SLEEP_INTERVAL,
        )
        self._training = LocalTrainingBackend(
            algorithm=self._algorithm,
            checkpoint_manager=self._checkpoint_manager,
            writer=self._writer,
        )

        self._total_connections = 0
        self._infer_wait_start: float | None = None

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
        # pending accumulators are only used to group results; drop them on shutdown
        self._pending_infer_results.clear()
        pending_futures, drained_requests = (
            await self._inference.abort_pending_requests(reason)
        )

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
        logger.info(
            f"Connection from {websocket.remote_address} opened. Total connections: {self._total_connections + 1}"
        )
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
                obs_list = unbatch_aggregate(obs, aggregate_method="concat")

                req_id = f"{session_id}-{uuid.uuid4()}"
                response_future = await self._inference.register_request(req_id)

                # put one queue entry per env; include expected_count to support
                # partial batching without completing the request early
                expected_count = len(env_ids)
                for i, (obs, eid) in enumerate(zip(obs_list, env_ids)):
                    infer_request = dict(
                        id=req_id,
                        obs=obs,
                        sub_index=i,
                        env_id=eid,
                        expected_count=expected_count,
                    )
                    await self._inference.queue.put(infer_request)

                try:
                    # response now may include env_ids and obs_list for per-env mapping
                    # response is expected to be a tuple:
                    # (action_arr, step_state_arr, resp_env_ids, resp_obs_list)
                    response = await response_future
                    action_arr, step_state_arr, resp_env_ids, resp_obs_list = response
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
                # process each env's feedback individually
                async with self._model_lock:
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

                        for key, value in log_dict.items():
                            self._writer.add_scalar(key, value, step)

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
                logger.info(
                    f"Connection from {websocket.remote_address} closed. Total connections: {self._total_connections}"
                )

    def should_infer(self) -> bool:
        current_qsize = self._inference.queue.qsize()
        now = time.monotonic()

        # allow queue size to be >= total connections to support per-connection
        # multi-env requests that expand into multiple queue entries

        infer_thereshold = self._mini_infer_batch_size or self._total_connections

        if (current_qsize >= infer_thereshold) and self._total_connections > 0:
            self._infer_wait_start = None
            return True

        if current_qsize > 0 and self._total_connections > 0:
            if self._infer_wait_start is None:
                self._infer_wait_start = now
            elif now - self._infer_wait_start > INFER_READY_TIMEOUT:
                logger.warning(
                    f"Infer queue has been waiting {now - self._infer_wait_start:.1f}s for {current_qsize}/{self._total_connections} environments"
                )
                self._infer_wait_start = now
        else:
            self._infer_wait_start = None

        return False

    async def _process_infer(self):
        batch = await self._inference.drain_batch()
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
            # action/runtime_state correspond to rows matching batch order
            # group indices by original request id so we can set a single
            # future.result per original request (which may have contained
            # multiple envs)
            indices_by_req = {}
            for idx, req in enumerate(batch):
                indices_by_req.setdefault(req["id"], []).append(idx)

            for req_id, indices in indices_by_req.items():
                first_req = batch[indices[0]]
                expected_count = int(first_req.get("expected_count", len(indices)))

                pending = self._pending_infer_results.get(req_id)
                if pending is None:
                    pending = {"expected_count": expected_count, "items": {}}
                    self._pending_infer_results[req_id] = pending
                elif int(pending["expected_count"]) != expected_count:
                    logger.warning(
                        f"Mismatched expected_count for req_id={req_id}: "
                        f"pending={pending['expected_count']}, incoming={expected_count}"
                    )

                # stash partial results keyed by sub_index
                for i in indices:
                    req = batch[i]
                    sub_index = int(req.get("sub_index"))
                    pending["items"][sub_index] = (
                        action[i],
                        slice_policy_step_state(step_state, slice(i, i + 1)),
                        req.get("env_id"),
                        req["obs"],
                    )

                exp = int(pending["expected_count"])
                if all(k in pending["items"] for k in range(exp)):
                    ordered = [pending["items"][k] for k in range(exp)]
                    act_result = np.stack([item[0] for item in ordered])
                    step_state_result = [item[1] for item in ordered]
                    # make this a numpy array so existing `(resp_env_ids == env_ids).all()` works reliably
                    env_id_list = np.asarray([item[2] for item in ordered])
                    obs_list = [item[3] for item in ordered]

                    future = self._inference.get_future(req_id)
                    if future and not future.done():
                        future.set_result(
                            (act_result, step_state_result, env_id_list, obs_list)
                        )
                    self._pending_infer_results.pop(req_id, None)

            for _ in batch:
                self._inference.queue.task_done()
        except Exception as exc:
            for req in batch:
                future = self._inference.get_future(req["id"])
                if future and not future.done():
                    future.set_exception(
                        RuntimeError(f"Inference processing error: {exc}")
                    )
            raise

    async def _run_control_cycle(self) -> None:
        async with self._model_lock:
            if self._training.should_learn():
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
