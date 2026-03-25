import asyncio
import signal
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import websockets.frames
from loguru import logger


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
        await abort_pending_infer_requests(self.shutdown_reason)

        if server is not None:
            if self.close_reason:
                await asyncio.gather(
                    *(
                        connection.close(
                            websockets.frames.CloseCode.GOING_AWAY,
                            self.close_reason,
                        )
                        for connection in server.connections
                    ),
                    return_exceptions=True,
                )
            server.close()
            await server.wait_closed()
            logger.info("WebSocket server closed.")

        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        logger.info("Scheduler task cancelled and cleaned up.")

        if extra_cleanup is not None:
            await extra_cleanup()
