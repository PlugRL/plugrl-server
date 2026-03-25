import asyncio
from collections import deque
from typing import Any
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
import ray
from torch.utils.tensorboard import SummaryWriter

from loguru import logger

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    MessageType,
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
)

from plugrl_server.algorithm.base_algorithm import DDPAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.common.data_utils import batch_aggregate
from plugrl_server.policy.state import slice_batched_state
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
from plugrl_server.server.training_backend import RayTrainingBackend


SCHEDULER_SLEEP_INTERVAL = 0.001  # seconds


class ServerStoppingError(RuntimeError):
    pass


class RayAgentServer:
    def __init__(
        self,
        inference_algorithm: DDPAlgorithm,
        checkpoint_manager: CheckpointManager,
        writer: SummaryWriter,
        learner_actor_ref: ray.ObjectRef,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm: DDPAlgorithm = inference_algorithm
        self._checkpoint_manager: CheckpointManager = checkpoint_manager
        self._writer = writer
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
        self._training = RayTrainingBackend(
            algorithm=self._algorithm,
            checkpoint_manager=self._checkpoint_manager,
            writer=self._writer,
            learner_actor_ref=self._learner_actor,
        )

        self._total_connections = 0

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
            extra_cleanup=self._training.shutdown,
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
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        session_id = str(websocket.remote_address)
        connection_counted = False

        try:
            await websocket.send(
                packer.pack(MetadataMessage(data=self._metadata).to_payload())
            )
            logger.info("Sent initial metadata to client.")
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

                obs, internal_state = infer_msg.data, None

                if not action_buffer:
                    req_id = f"{session_id}-{uuid.uuid4()}"
                    response_future = await self._inference.register_request(req_id)

                    infer_request = dict(id=req_id, obs=obs)
                    await self._inference.queue.put(infer_request)

                    try:
                        action, internal_state = await response_future
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

                if self._algorithm.break_action_chunk:
                    action = action_buffer.popleft()
                else:
                    action = np.array(
                        [action_buffer.popleft() for _ in range(len(action_buffer))]
                    )

                action_response = ActionMessage(
                    data=dict(action=action)
                )
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
                    await asyncio.to_thread(self._training.log, log_dict, step=step)

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
        return (
            self._inference.queue.qsize() >= self._total_connections // 2
            and self._total_connections > 0
        )

    async def _process_infer(self):
        batch = await self._inference.drain_batch()
        if not batch:
            return
        try:
            obs = batch_aggregate([req["obs"] for req in batch])
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, step_state = self._algorithm.infer_step(
                    obs,
                    include_train_state=False,
                )
                internal_state = step_state.runtime_state
            logger.debug(f"Inference done for batch size {len(batch)}")
            for i, req in enumerate(batch):
                self._inference.resolve_request(
                    req,
                    result=(
                        action[i : i + 1],
                        slice_batched_state(internal_state, slice(i, i + 1)),
                    ),
                )
        except Exception as exc:
            self._inference.fail_batch(batch, exc)
            raise

    async def _run_control_cycle(self) -> None:
        async with self._model_lock:
            if self._training.should_learn():
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
