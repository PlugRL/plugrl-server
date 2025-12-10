import asyncio
import traceback
import time
from collections import deque
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
import ray
from torch.utils.tensorboard import SummaryWriter

from loguru import logger

from vlarl_client import msgpack_numpy
from vlarl_client.websocket_worker_agent import MessageType

from vlarl_launcher.algorithm.base_algorithm import DDPAlgorithm
from vlarl_launcher.common.checkpoint_manager import CheckpointManager, Checkpoint 
from vlarl_launcher.common.data_utils import batch_aggregate
from vlarl_launcher.server.ray_learner import LearnerActor

import swanlab
import wandb

SCHEDULER_SLEEP_INTERVAL = 0.001  # seconds

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
        self._learner_actor: LearnerActor = learner_actor_ref
        
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        
        self._infer_queue = asyncio.Queue()
        self._response_futures = {}
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._server = None
        
        self._total_connections = 0
        self._fatal_reported = False
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())
    
    async def run(self):
        scheduler_task = asyncio.create_task(self._main_scheduler_loop())
        try:
            async with _server.serve(
                self._handler, self._host, self._port, compression=None, max_size=None
            ) as server:
                self._server = server
                logger.info(f"Agent Server is listening on {self._host}:{self._port}")
                await self._stop_event.wait()
            await scheduler_task
            logger.info("Scheduler task completed.")
        except Exception as exc:
            await self._handle_fatal("agent server runtime", exc)
            raise
        finally:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                logger.info("WebSocket server closed.")

    async def _handle_fatal(self, context: str, exc: BaseException) -> None:
        if not self._fatal_reported:
            traceback_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.error(f"Fatal error in {context}:\n{traceback_str}")
            self._fatal_reported = True
        self._stop_event.set()
        await self._shutdown_ray()

    async def _shutdown_ray(self) -> None:
        if ray.is_initialized():
            await asyncio.to_thread(ray.shutdown)

    def _resolve_infer_future(self, request: dict, *, result=None, exc: BaseException | None = None) -> None:
        future = self._response_futures.get(request["id"])
        if future and not future.done():
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(result)
        self._infer_queue.task_done()

    def _fail_infer_batch(self, batch: list[dict], exc: Exception) -> None:
        for req in batch:
            wrapped_exc = RuntimeError(f"Inference processing error: {exc}")
            self._resolve_infer_future(req, exc=wrapped_exc)
            
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        session_id = str(websocket.remote_address)

        try:
            metadata_message = dict(message_type=str(MessageType.METADATA), data=self._metadata)
            await websocket.send(packer.pack(metadata_message))
            logger.info("Sent initial metadata to client.")
            self._total_connections += 1
            prev_node: tuple = (-1, "")
            terminated, truncated = False, False  
            action_buffer = deque()
            while True:
                packed_infer_msg = await websocket.recv()
                infer_msg = msgpack_numpy.unpackb(packed_infer_msg)
                
                obs, internal_state = infer_msg.get("data"), None
                
                if not action_buffer:
                    req_id = f"{session_id}-{uuid.uuid4()}"
                    response_future = asyncio.Future()

                    async with self._lock:
                        self._response_futures[req_id] = response_future

                    infer_request = dict(id=req_id, obs=obs)
                    await self._infer_queue.put(infer_request)
                    
                    action, internal_state = await response_future
                
                    async with self._lock:
                        self._response_futures.pop(req_id, None)
                        
                    action_buffer.extend(action.swapaxes(1, 0))
                
                if self._algorithm.break_action_chunk:
                    action = action_buffer.popleft()
                else:
                    action = np.array([action_buffer.popleft() for _ in range(len(action_buffer))])
                
                action_response = dict(message_type=str(MessageType.ACTION), data=dict(action=action))
                await websocket.send(packer.pack(action_response))
                    
                packed_feedback_msg = await websocket.recv()
                feedback_msg = msgpack_numpy.unpackb(packed_feedback_msg)
                
                if feedback_msg.get("message_type") != str(MessageType.FEEDBACK):
                    logger.warning(f"Expected a FEEDBACK message but received: {feedback_msg.get('message_type')}")
                    continue
                
                feedback_data = feedback_msg.get("data")

                next_obs, reward, next_terminated, next_truncated, info = feedback_data.values()
                
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
                        prev_node=prev_node
                    )
                    await asyncio.to_thread(self.log, log_dict, step=step)
                    
                terminated, truncated = next_terminated, next_truncated
                
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed.")
            self._total_connections -= 1
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Internal server error:\n{traceback_str}")
            self._total_connections -= 1
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error."
            )
            
    def log(self, log_dict: dict, step: int):
        for key, value in log_dict.items():
            self._writer.add_scalar(key, value, step)

    def should_infer(self) -> bool:
        return self._infer_queue.qsize() >= self._total_connections // 2 and self._total_connections > 0
    
    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    async def _update_inference_policy(self, checkpoint: Checkpoint):
        self._algorithm.load_learner_state(checkpoint)
        logger.info("Local inference policy successfully updated.")

    async def _process_infer(self):
        batch = [await self._infer_queue.get() for _ in range(self._infer_queue.qsize())]
        if not batch:
            return
        try:
            obs = batch_aggregate([req["obs"] for req in batch])
            logger.debug(f"Processing inference for batch size {len(batch)}")
            async with self._model_lock:
                action, internal_state = self._algorithm.infer(obs)
            logger.debug(f"Inference done for batch size {len(batch)}")
            for i, req in enumerate(batch):
                self._resolve_infer_future(req, result=(action[i:i+1], internal_state[i:i+1]))
        except Exception as exc:
            self._fail_infer_batch(batch, exc)
            raise

    async def _process_learn(self):
        logger.info("Scheduler initiating distributed training via LearnerActor.")
        self._algorithm.pre_learn()
        global_step, meta_info, serializable_buffer_data = self._algorithm.get_server_data()
        buffer_data_ref = await asyncio.to_thread(ray.put, serializable_buffer_data)

        learn_ref = self._learner_actor.learn.remote(global_step, meta_info, buffer_data_ref)
        checkpoint, global_step, train_info = await asyncio.to_thread(ray.get, learn_ref)
        
        await self._update_inference_policy(checkpoint)

        await asyncio.to_thread(self.log, train_info, step=global_step)
        logger.info(f"Logged training info for step {global_step}.")
        self._algorithm.post_learn()

    async def _process_save(self):
        checkpoint = await asyncio.to_thread(self._algorithm.create_ddp_checkpoint) 
        await asyncio.to_thread(self._checkpoint_manager.save_checkpoint, checkpoint)
        logger.info(f"Checkpoint saved locally at step {checkpoint.step}.")

    async def _main_scheduler_loop(self):
        try:
            while not self._stop_event.is_set():
                if self.should_infer():
                    await self._process_infer()

                async with self._model_lock:
                    if self.should_learn():
                        await self._process_learn()

                    stop_requested = self.should_stop()
                    if self.should_save() or stop_requested:
                        await self._process_save()

                    if stop_requested:
                        self._stop_event.set()
                        logger.info("Stopping server as the algorithm signaled to stop.")
                        await self._shutdown_ray()
                        break

                await asyncio.sleep(SCHEDULER_SLEEP_INTERVAL)
        except Exception as exc:
            await self._handle_fatal("scheduler loop", exc)
            raise