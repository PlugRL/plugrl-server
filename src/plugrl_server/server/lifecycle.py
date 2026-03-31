import asyncio
import signal
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import websockets.frames
from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)
CONNECTION_CLOSE_TIMEOUT = 2.0
SERVER_CLOSE_TIMEOUT = 2.0


class ServerLifecycle:
    def __init__(self) -> None:
        self.stop_event = asyncio.Event()
        self.close_reason = ""
        self.shutdown_reason = "Server is shutting down."
        self.fatal_reported = False

    def request_shutdown(self, reason: str, close_reason: str | None = None) -> None:
        if close_reason is not None:
            self.close_reason = close_reason
        if self.stop_event.is_set():
            return
        self.shutdown_reason = reason
        logger.info(reason)
        self.stop_event.set()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda sig=sig: self.request_shutdown(
                        f"Received {signal.Signals(sig).name}. Closing connections and allowing workers to reconnect."
                    ),
                )
            except (NotImplementedError, RuntimeError):
                continue

    async def handle_fatal(
        self,
        context: str,
        exc: BaseException,
        *,
        extra_cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if not self.fatal_reported:
            traceback_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            logger.error(f"Fatal error in {context}:\n{traceback_str}")
            self.fatal_reported = True
        self.close_reason = ""
        self.shutdown_reason = f"Fatal error in {context}."
        self.stop_event.set()
        if extra_cleanup is not None:
            await extra_cleanup()

    async def shutdown(
        self,
        *,
        scheduler_task: asyncio.Task,
        server: Any,
        abort_pending_infer_requests: Callable[[str], Awaitable[None]],
        extra_cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.stop_event.set()
        logger.info("Shutdown started: aborting pending infer requests.")
        await abort_pending_infer_requests(self.shutdown_reason)

        if server is not None:
            close_reason = self.close_reason or "Server shutdown"
            logger.info("Shutdown closing %s websocket connection(s).", len(server.connections))
            await asyncio.gather(
                *(
                    _close_connection_with_timeout(
                        connection,
                        code=websockets.frames.CloseCode.GOING_AWAY,
                        reason=close_reason,
                    )
                    for connection in list(server.connections)
                ),
                return_exceptions=True,
            )
            for connection in list(server.connections):
                transport = getattr(connection, "transport", None)
                if transport is not None:
                    transport.abort()
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=SERVER_CLOSE_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for websocket server to close.")
            logger.info("WebSocket server closed.")

        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        logger.info("Scheduler task cancelled and cleaned up.")

        if extra_cleanup is not None:
            await extra_cleanup()


async def _close_connection_with_timeout(connection: Any, *, code: int, reason: str) -> None:
    try:
        await asyncio.wait_for(
            connection.close(code, reason),
            timeout=CONNECTION_CLOSE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Timed out closing websocket connection; aborting transport.")
        transport = getattr(connection, "transport", None)
        if transport is not None:
            transport.abort()
