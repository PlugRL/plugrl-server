import asyncio
from collections.abc import Awaitable, Callable


class RuntimeScheduler:
    """The server's main loop: infer when a batch is ready, then learn or save.

    Between iterations it used to sleep `sleep_interval`, 0.1 ms. On Linux
    asyncio waits in epoll_wait, whose timeout has millisecond resolution, and
    Python rounds a positive timeout up to 1 ms. So an infer queued while that
    sleep was pending, with no further network event to wake the loop, waited
    up to a millisecond to be served: 1.3 ms of a 2.2 ms round trip on one
    machine (E46, results/explore.txt).

    Given `wake`, the loop waits on it instead. Whoever changes what the loop
    decides on sets it: an infer being queued, a feedback being processed, a
    connection opening or closing, a stop being requested. `idle_timeout`
    bounds the wait, so that the loop still comes round without one.
    Without `wake` it sleeps as before.
    """

    def __init__(
        self,
        *,
        stop_event: asyncio.Event,
        sleep_interval: float,
        wake: asyncio.Event | None = None,
        idle_timeout: float = 0.05,
    ) -> None:
        self._stop_event = stop_event
        self._sleep_interval = sleep_interval
        self._wake = wake
        self._idle_timeout = idle_timeout

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
            if self._wake is None:
                await asyncio.sleep(self._sleep_interval)
                continue
            if should_infer():
                continue  # work arrived while this iteration ran
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._idle_timeout)
            except TimeoutError:
                pass
            # Nothing runs between the wait returning and this clear, so a
            # wake set from now on is seen by the next wait.
            self._wake.clear()
