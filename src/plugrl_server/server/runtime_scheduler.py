import asyncio
from collections.abc import Awaitable, Callable


class RuntimeScheduler:
    def __init__(self, *, stop_event: asyncio.Event, sleep_interval: float) -> None:
        self._stop_event = stop_event
        self._sleep_interval = sleep_interval

    async def run(
        self,
        *,
        should_infer: Callable[[], bool],
        process_infer: Callable[[], Awaitable[None]],
        run_control_cycle: Callable[[], Awaitable[None]],
    ) -> None:
        while not self._stop_event.is_set():
            if should_infer():
                await process_infer()

            await run_control_cycle()
            await asyncio.sleep(self._sleep_interval)
