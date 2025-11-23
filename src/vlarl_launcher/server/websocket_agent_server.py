import asyncio
import traceback
import time
from collections import deque
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames
import uuid
from loguru import logger
import swanlab
import wandb

from vlarl_client import msgpack_numpy
from vlarl_client.websocket_worker_agent import MessageType

from vlarl_launcher.algorithm.base_algorithm import BaseAlgorithm
from vlarl_launcher.common.checkpoint_manager import CheckpointManager
from vlarl_launcher.common.data_utils import batch_aggregate

SCHEDULER_SLEEP_INTERVAL = 0.001  # seconds

class WebSocketAgentServer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        tracker: swanlab.run.SwanLabRun | wandb.Run,
        host: str = "0.0.0.0", 
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._tracker = tracker
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        
        self._infer_queue = asyncio.Queue()
        self._response_futures = {}
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._server = None
        self._stop_event = asyncio.Event()
        
        self._total_connections = 0
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())
    
    async def run(self):
        scheduler_task = asyncio.create_task(self._main_scheduler_loop())
        try:
            async with _server.serve(
                self._handler, self._host, self._port, compression=None, max_size=None
            ) as server:
                logger.info(f"Agent Server is listening on {self._host}:{self._port}")
                self._server = server
                await self._stop_event.wait()
        finally:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                logger.info("WebSocket server closed.")

            scheduler_task.cancel()
            await asyncio.gather(scheduler_task, return_exceptions=True)
            logger.info("Scheduler task cancelled and cleaned up.")
            
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
                    self._tracker.log(log_dict, step=step)
                    
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

    def should_infer(self) -> bool:
        return self._infer_queue.qsize() == self._total_connections and self._total_connections > 0
    
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
                    future.set_result((action[i:i+1], internal_state[i:i+1]))
                self._infer_queue.task_done()
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Error during inference processing:\n{traceback_str}")
            for req in batch:
                future = self._response_futures.get(req["id"])
                if future and not future.done():
                    future.set_exception(Exception("Inference processing error."))

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    async def _process_learn(self):
        try:
            self._algorithm.pre_learn()
            step, log_dict = await asyncio.to_thread(self._algorithm.learn)
            self._algorithm.post_learn()
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Error during learning processing:\n{traceback_str}")
            return
        self._tracker.log(log_dict, step=step)
            
    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    def should_save(self) -> bool:
        return self._algorithm.should_save()
    
    async def _process_save(self):
        try: 
            checkpoint = self._algorithm.create_checkpoint()
            self._checkpoint_manager.save_checkpoint(checkpoint)
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Error during saving checkpoint:\n{traceback_str}")

    async def _main_scheduler_loop(self):
        try:
            while True:
                if self.should_infer():
                    await self._process_infer()
                    
                async with self._model_lock:
                    if self.should_learn():
                        await self._process_learn()
                    
                async with self._model_lock:
                    if self.should_save() or self.should_stop():
                        await self._process_save()
                        
                if self.should_stop():
                    self._stop_event.set()
                    logger.info("Stopping server as the algorithm signaled to stop.")
                    break
                    
                await asyncio.sleep(SCHEDULER_SLEEP_INTERVAL)
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Error in main scheduler loop:\n{traceback_str}")