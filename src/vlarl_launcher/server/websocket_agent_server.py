import asyncio
import traceback
import numpy as np

import websockets.asyncio.server as _server
import websockets.frames
from loguru import logger

from vlarl_client import msgpack_numpy
from vlarl_client.websocket_worker_agent import MessageType

from vlarl_launcher.algorithm.base_algorithm import BaseAlgorithm

class WebSocketAgentServer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        host: str = "0.0.0.0", 
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._algorithm = algorithm
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        
    def serve_forever(self) -> None:
        asyncio.run(self.run())
        
    async def run(self):
        async with _server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            logger.info(f"Agent Server is listening on {self._host}:{self._port}")
            await server.serve_forever()
            
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        
        try:
            metadata_message = dict(message_type=str(MessageType.METADATA), data=self._metadata)
            await websocket.send(packer.pack(metadata_message))
            logger.info("Sent initial metadata to client.")

            while True:
                packed_infer_msg = await websocket.recv()
                infer_msg = msgpack_numpy.unpackb(packed_infer_msg)
                
                if infer_msg.get("message_type") != str(MessageType.INFER):
                    logger.warning(f"Expected an INFER message but received: {infer_msg.get('message_type')}")
                    continue

                obs = infer_msg.get("data")

                action_dict = self._algorithm.infer(obs)
                action_response = dict(message_type=str(MessageType.ACTION), data=action_dict)
                await websocket.send(packer.pack(action_response))

                packed_feedback_msg = await websocket.recv()
                feedback_msg = msgpack_numpy.unpackb(packed_feedback_msg)

                if feedback_msg.get("message_type") == str(MessageType.FEEDBACK):
                    feedback_data = feedback_msg.get("data")
                    obs, reward, terminated, truncated, info = feedback_data.values()
                    self._algorithm.feedback(obs, reward, terminated, truncated, info)
                else:
                    logger.warning(f"Expected a FEEDBACK message but received: {feedback_msg.get('message_type')}")
        
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed.")
        except Exception:
            traceback_str = traceback.format_exc()
            logger.error(f"Internal server error:\n{traceback_str}")
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error."
            )