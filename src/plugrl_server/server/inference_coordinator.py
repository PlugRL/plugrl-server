import asyncio
from typing import Any


class InferenceCoordinator:
    def __init__(self, *, stopping_error_factory) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: dict[str, asyncio.Future] = {}
        self.lock = asyncio.Lock()
        self._stopping_error_factory = stopping_error_factory

    async def register_request(self, request_id: str) -> asyncio.Future:
        response_future = asyncio.Future()
        async with self.lock:
            self.response_futures[request_id] = response_future
        return response_future

    async def pop_request(self, request_id: str) -> None:
        async with self.lock:
            self.response_futures.pop(request_id, None)

    async def abort_pending_requests(self, reason: str) -> tuple[int, int]:
        async with self.lock:
            pending_futures = 0
            for future in self.response_futures.values():
                if not future.done():
                    future.set_exception(self._stopping_error_factory(reason))
                    pending_futures += 1
            self.response_futures.clear()

        drained_requests = 0
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.queue.task_done()
                drained_requests += 1

        return pending_futures, drained_requests

    async def drain_batch(self) -> list[dict[str, Any]]:
        batch = []
        for _ in range(self.queue.qsize()):
            batch.append(await self.queue.get())
        return batch

    def get_future(self, request_id: str) -> asyncio.Future | None:
        return self.response_futures.get(request_id)

    def resolve_request(
        self,
        request: dict[str, Any],
        *,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        future = self.response_futures.get(request["id"])
        if future and not future.done():
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(result)
        self.queue.task_done()

    def fail_batch(self, batch: list[dict[str, Any]], exc: Exception) -> None:
        for request in batch:
            wrapped_exc = RuntimeError(f"Inference processing error: {exc}")
            self.resolve_request(request, exc=wrapped_exc)
