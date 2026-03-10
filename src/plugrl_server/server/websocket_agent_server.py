import asyncio
import signal
import traceback
import time
from collections import deque
from typing import Any
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from plugrl_client import msgpack_numpy
from plugrl_client.websocket_worker_agent import MessageType, SERVER_STOP_REASON, SERVER_RESYNC_REASON

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.common.data_utils import batch_aggregate

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
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._server: Any = None
        self._stop_event = asyncio.Event()

        self._total_connections = 0
        self._infer_wait_start: float | None = None
        self._close_reason = ""
        self._shutdown_reason = "Server is shutting down."
        self._fatal_reported = False

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
            prev_node: tuple = (-1, "")
            terminated, truncated = False, False
            action_buffer = deque()
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

                obs, internal_state = infer_msg.get("data"), None

                if not action_buffer:
                    req_id = f"{session_id}-{uuid.uuid4()}"
                    response_future = asyncio.Future()
                    async with self._lock:
                        self._response_futures[req_id] = response_future
                    infer_request = dict(id=req_id, obs=obs)
                    await self._infer_queue.put(infer_request)

                    try:
                        action, internal_state = await response_future
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

                    action_buffer.extend(action.swapaxes(1, 0))

                if self._algorithm.break_action_chunk:
                    action = action_buffer.popleft()
                else:
                    action = np.array(
                        [action_buffer.popleft() for _ in range(len(action_buffer))]
                    )

                action_response = dict(
                    message_type=str(MessageType.ACTION), data=dict(action=action)
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

                next_obs, reward, next_terminated, next_truncated, info = (
                    feedback_data.values()
                )
                async with self._model_lock:
                    prev_node, step, log_dict = self._algorithm.feedback(
                        obs=obs,
                        internal_state=internal_state,
                        terminated=terminated,
                        truncated=truncated,
                        next_obs=next_obs,
                        reward=reward,
                        next_terminated=next_terminated,
                        next_truncated=next_truncated,
                        info=info,
                        prev_node=prev_node,
                    )
                    for key, value in log_dict.items():
                        self._writer.add_scalar(key, value, step)

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
                logger.info(
                    f"Connection from {websocket.remote_address} closed. Total connections: {self._total_connections}"
                )

    def should_infer(self) -> bool:
        current_qsize = self._infer_queue.qsize()
        now = time.monotonic()

        if current_qsize == self._total_connections and self._total_connections > 0:
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
            obs = batch_aggregate([req["obs"] for req in batch])
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, internal_state = self._algorithm.infer(obs)
            logger.debug(f"Inference done for batch size {len(batch)}")
            for i, req in enumerate(batch):
                future = self._response_futures.get(req["id"])
                if future and not future.done():
                    future.set_result((action[i : i + 1], internal_state[i : i + 1]))
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
