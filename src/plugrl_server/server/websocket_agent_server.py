import asyncio
import signal
import traceback
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

        self._infer_queue = asyncio.Queue()
        self._response_futures = {}
        # Accumulate partial inference outputs per req_id; only fulfill the
        # corresponding future once all sub-indices for that req_id arrive.
        self._pending_infer_results: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._server: Any = None
        self._stop_event = asyncio.Event()

        self._total_connections = 0
        self._infer_wait_start: float | None = None
        self._close_reason = ""
        self._shutdown_reason = "Server is shutting down."
        self._fatal_reported = False

        self._mini_infer_batch_size = mini_infer_batch_size

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _request_shutdown(self, reason: str, close_reason: str | None = None) -> None:
        if close_reason is not None:
            self._close_reason = close_reason
        if self._stop_event.is_set():
            return
        self._shutdown_reason = reason
        logger.info(reason)
        self._stop_event.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda sig=sig: self._request_shutdown(
                        f"Received {signal.Signals(sig).name}. Closing connections and allowing workers to reconnect."
                    ),
                )
            except (NotImplementedError, RuntimeError):
                continue

    async def _handle_fatal(self, context: str, exc: BaseException) -> None:
        if not self._fatal_reported:
            traceback_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            logger.error(f"Fatal error in {context}:\n{traceback_str}")
            self._fatal_reported = True
        self._close_reason = ""
        self._shutdown_reason = f"Fatal error in {context}."
        self._stop_event.set()

    async def _abort_pending_infer_requests(self, reason: str) -> None:
        async with self._lock:
            pending_futures = 0
            for future in self._response_futures.values():
                if not future.done():
                    future.set_exception(ServerStoppingError(reason))
                    pending_futures += 1
            self._response_futures.clear()

        # pending accumulators are only used to group results; drop them on shutdown
        self._pending_infer_results.clear()

        drained_requests = 0
        while True:
            try:
                self._infer_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._infer_queue.task_done()
                drained_requests += 1

        if pending_futures > 0 or drained_requests > 0:
            logger.info(
                "Shutdown cleanup finished: "
                f"pending_futures={pending_futures}, drained_requests={drained_requests}"
            )

    async def _shutdown(self, scheduler_task: asyncio.Task, reason: str) -> None:
        self._stop_event.set()
        await self._abort_pending_infer_requests(reason)

        if self._server is not None:
            if self._close_reason:
                await asyncio.gather(
                    *(
                        connection.close(
                            websockets.frames.CloseCode.GOING_AWAY,
                            self._close_reason,
                        )
                        for connection in self._server.connections
                    ),
                    return_exceptions=True,
                )
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("WebSocket server closed.")

        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        logger.info("Scheduler task cancelled and cleaned up.")

    async def run(self):
        scheduler_task = asyncio.create_task(self._main_scheduler_loop())
        self._install_signal_handlers()
        try:
            self._server = await _server.serve(
                self._handler, self._host, self._port, compression=None, max_size=None
            )
            logger.info(f"Agent Server is listening on {self._host}:{self._port}")
            await self._stop_event.wait()
        except asyncio.CancelledError:
            self._shutdown_reason = "Server run task was cancelled."
            logger.info(
                "Server cancellation received. Closing connections and allowing workers to reconnect."
            )
            raise
        except Exception as exc:
            await self._handle_fatal("agent server runtime", exc)
            raise
        finally:
            await asyncio.shield(self._shutdown(scheduler_task, self._shutdown_reason))

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(
            f"Connection from {websocket.remote_address} opened. Total connections: {self._total_connections + 1}"
        )
        packer = msgpack_numpy.Packer()
        session_id = str(websocket.remote_address)
        connection_counted = False

        try:
            metadata_message = dict(
                message_type=str(MessageType.METADATA), data=self._metadata
            )
            await websocket.send(packer.pack(metadata_message))
            self._total_connections += 1
            connection_counted = True
            prev_node_map: dict = {}
            internal_state_map: dict = {}
            terminated_map: dict = {}
            truncated_map: dict = {}
            last_obs_map: dict = {}
            while True:
                packed_infer_msg = await websocket.recv()
                infer_msg = msgpack_numpy.unpackb(packed_infer_msg)
                if infer_msg.get("message_type") != str(MessageType.INFER):
                    logger.warning(
                        "Expected an INFER message but received "
                        f"{infer_msg.get('message_type')}. Requesting worker resync."
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.GOING_AWAY,
                        reason=SERVER_RESYNC_REASON,
                    )
                    break

                obs, env_ids, internal_state = (
                    infer_msg.get("data"),
                    infer_msg.get("env_indices"),
                    None,
                )
                obs_list = unbatch_aggregate(obs, aggregate_method="concat")

                req_id = f"{session_id}-{uuid.uuid4()}"
                response_future = asyncio.Future()
                async with self._lock:
                    self._response_futures[req_id] = response_future

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
                    await self._infer_queue.put(infer_request)

                try:
                    # response now may include env_ids and obs_list for per-env mapping
                    # response is expected to be a tuple: (action_arr, internal_state_arr, resp_env_ids, resp_obs_list)
                    response = await response_future
                    action_arr, internal_state_arr, resp_env_ids, resp_obs_list = (
                        response
                    )
                    assert np.array_equal(
                        np.asarray(resp_env_ids), np.asarray(env_ids)
                    ), "Response env_ids do not match request env_ids"
                except ServerStoppingError:
                    async with self._lock:
                        self._response_futures.pop(req_id, None)
                    if self._close_reason:
                        try:
                            await websocket.close(
                                code=websockets.frames.CloseCode.GOING_AWAY,
                                reason=self._close_reason,
                            )
                        except Exception:
                            pass
                    logger.info(
                        "Shutdown interrupted an in-flight inference request "
                        f"from {websocket.remote_address}."
                    )
                    break
                else:
                    async with self._lock:
                        self._response_futures.pop(req_id, None)
                # register per-env internal states and last observations
                for i, eid in enumerate(resp_env_ids):
                    internal_state_map[eid] = internal_state_arr[i]
                    last_obs_map[eid] = resp_obs_list[i]

                action_response = dict(
                    message_type=str(MessageType.ACTION),
                    data=dict(env_ids=resp_env_ids, action=action_arr.swapaxes(0, 1)),
                )
                await websocket.send(packer.pack(action_response))

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
                feedback_msg = msgpack_numpy.unpackb(packed_feedback_msg)

                if feedback_msg.get("message_type") != str(MessageType.FEEDBACK):
                    logger.warning(
                        "Expected a FEEDBACK message but received: "
                        f"{feedback_msg.get('message_type')}. Requesting worker resync."
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.GOING_AWAY,
                        reason=SERVER_RESYNC_REASON,
                    )
                    break

                feedback_data = feedback_msg.get("data")

                fb_env_ids, _ = (
                    feedback_msg.get("env_indices"),
                    feedback_msg.get("step_ids"),
                )
                # assume feedback_data contains parallel lists/arrays with an 'env_ids' field
                (
                    next_obs_batch,
                    reward_list,
                    next_terminated_list,
                    next_truncated_list,
                    info_batch,
                ) = feedback_data.values()
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
                        internal_state = internal_state_map.get(eid, None)
                        terminated = terminated_map.get(eid, False)
                        truncated = truncated_map.get(eid, False)
                        last_obs = last_obs_map.get(eid, {})
                        prev_node_res, step, log_dict = self._algorithm.feedback(
                            obs=last_obs,
                            internal_state=internal_state,
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
        current_qsize = self._infer_queue.qsize()
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
        batch = []
        for _ in range(self._infer_queue.qsize()):
            batch.append(await self._infer_queue.get())
        try:
            obs = batch_aggregate(
                [req["obs"] for req in batch], aggregate_method="concat"
            )
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, internal_state = self._algorithm.infer(obs)
            logger.debug(f"Inference done for batch size {len(batch)}")
            # action/internal_state correspond to rows matching batch order
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
                        internal_state[i : i + 1],
                        req.get("env_id"),
                        req["obs"],
                    )

                exp = int(pending["expected_count"])
                if all(k in pending["items"] for k in range(exp)):
                    ordered = [pending["items"][k] for k in range(exp)]
                    act_result = np.stack([item[0] for item in ordered])
                    int_result = [item[1] for item in ordered]
                    # make this a numpy array so existing `(resp_env_ids == env_ids).all()` works reliably
                    env_id_list = np.asarray([item[2] for item in ordered])
                    obs_list = [item[3] for item in ordered]

                    future = self._response_futures.get(req_id)
                    if future and not future.done():
                        future.set_result(
                            (act_result, int_result, env_id_list, obs_list)
                        )
                    self._pending_infer_results.pop(req_id, None)

            for _ in batch:
                self._infer_queue.task_done()
        except Exception as exc:
            for req in batch:
                future = self._response_futures.get(req["id"])
                if future and not future.done():
                    future.set_exception(
                        RuntimeError(f"Inference processing error: {exc}")
                    )
            raise

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    async def _process_learn(self):
        self._algorithm.pre_learn()
        step, log_dict = await asyncio.to_thread(self._algorithm.learn)
        self._algorithm.post_learn()
        for key, value in log_dict.items():
            self._writer.add_scalar(key, value, step)

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    async def _process_save(self):
        checkpoint = self._algorithm.create_checkpoint()
        self._checkpoint_manager.save_checkpoint(checkpoint)

    async def _main_scheduler_loop(self):
        try:
            while not self._stop_event.is_set():
                if self.should_infer():
                    await self._process_infer()

                async with self._model_lock:
                    if self.should_learn():
                        await self._process_learn()

                async with self._model_lock:
                    if self.should_save() or self.should_stop():
                        await self._process_save()

                if self.should_stop():
                    self._request_shutdown(
                        "Stopping server as the algorithm signaled to stop.",
                        close_reason=SERVER_STOP_REASON,
                    )
                    break

                await asyncio.sleep(SCHEDULER_SLEEP_INTERVAL)
        except Exception as exc:
            await self._handle_fatal("main scheduler loop", exc)
            raise
