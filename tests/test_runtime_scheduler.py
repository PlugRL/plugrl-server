"""The scheduler serves an infer when it is queued, not when a timer next fires.

It used to sleep 0.1 ms between iterations. On Linux asyncio waits in
epoll_wait, whose timeout has millisecond resolution, so the sleep became a
wait of up to 1 ms whenever no network event followed: 1.3 ms of a 2.2 ms
round trip on one machine (E46). Now the server sets an event whenever there
is something to decide, and the scheduler waits on that.
"""

from __future__ import annotations

import asyncio
import time

from plugrl_server.server.runtime_scheduler import RuntimeScheduler


class _Server:
    """One queued request at a time; records when each is served."""

    def __init__(self) -> None:
        self.queued = 0
        self.served: list[float] = []

    def should_infer(self) -> bool:
        return self.queued > 0

    async def process_infer(self) -> None:
        self.queued = 0
        self.served.append(time.perf_counter())

    async def run_control_cycle(self) -> None:
        pass


async def _run(scheduler: RuntimeScheduler, server: _Server, body) -> None:
    task = asyncio.create_task(
        scheduler.run(
            should_infer=server.should_infer,
            process_infer=server.process_infer,
            run_control_cycle=server.run_control_cycle,
        )
    )
    try:
        await body()
    finally:
        scheduler._stop_event.set()
        if scheduler._wake is not None:
            scheduler._wake.set()
        await asyncio.wait_for(task, timeout=5)


def test_a_queued_infer_is_served_at_once_not_at_the_idle_timeout():
    async def main():
        stop, wake = asyncio.Event(), asyncio.Event()
        scheduler = RuntimeScheduler(
            stop_event=stop, sleep_interval=0.0001, wake=wake, idle_timeout=30.0
        )
        server = _Server()
        waits = []

        async def body():
            await asyncio.sleep(0.05)  # the scheduler is now waiting on `wake`
            for _ in range(20):
                queued_at = time.perf_counter()
                server.queued = 1
                wake.set()
                while len(server.served) <= len(waits):
                    await asyncio.sleep(0)
                waits.append(server.served[-1] - queued_at)

        await _run(scheduler, server, body)
        return waits

    waits = asyncio.run(main())

    # Against an idle timeout of 30 s, and the old sleep's 1 ms.
    assert max(waits) < 0.5
    assert sorted(waits)[len(waits) // 2] < 0.001


def test_a_wake_while_the_loop_is_busy_is_not_lost():
    """Set during the control cycle, it must still end the next wait."""

    async def main():
        stop, wake = asyncio.Event(), asyncio.Event()
        scheduler = RuntimeScheduler(
            stop_event=stop, sleep_interval=0.0001, wake=wake, idle_timeout=30.0
        )
        server = _Server()
        cycles = 0

        async def control_cycle():
            nonlocal cycles
            cycles += 1
            if cycles == 2:
                # Work arrives while the loop is busy, as a feedback does.
                server.queued = 1
                wake.set()

        server.run_control_cycle = control_cycle

        async def body():
            await asyncio.sleep(0.05)
            wake.set()  # one iteration: its control cycle queues an infer
            started = time.perf_counter()
            while not server.served:
                await asyncio.sleep(0)
                assert time.perf_counter() - started < 5, "the wake was lost"

        await _run(scheduler, server, body)
        return server.served

    assert len(asyncio.run(main())) == 1


def test_without_wake_it_sleeps_as_before():
    async def main():
        stop = asyncio.Event()
        scheduler = RuntimeScheduler(stop_event=stop, sleep_interval=0.0001)
        server = _Server()

        async def body():
            server.queued = 1
            started = time.perf_counter()
            while not server.served:
                await asyncio.sleep(0)
                assert time.perf_counter() - started < 5

        await _run(scheduler, server, body)
        return server.served

    assert len(asyncio.run(main())) == 1
