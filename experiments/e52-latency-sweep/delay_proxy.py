"""A TCP proxy that holds every byte for D ms in each direction.

    python delay_proxy.py --listen 8821 --upstream 8820 --delay-ms 5

A client that connects to --listen reaches 127.0.0.1:--upstream, with each
chunk it sends, and each chunk it receives, forwarded D ms after it arrived,
in order. A request and its answer therefore take 2 D longer, which is what a
link with a one-way latency of D does to a round trip. Bandwidth is not
limited. With --delay-ms 0 it only forwards, which measures its own cost.
"""

from __future__ import annotations

import argparse
import asyncio
import time


async def pump(reader, writer, delay: float) -> None:
    queue: asyncio.Queue = asyncio.Queue()

    async def read() -> None:
        while data := await reader.read(1 << 20):
            queue.put_nowait((time.monotonic() + delay, data))
        queue.put_nowait(None)

    async def write() -> None:
        while (item := await queue.get()) is not None:
            due, data = item
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(data)
            await writer.drain()
        writer.close()

    await asyncio.gather(read(), write())


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", type=int, required=True)
    p.add_argument("--delay-ms", type=float, required=True)
    a = p.parse_args()
    delay = a.delay_ms / 1000.0

    async def handle(client_reader, client_writer) -> None:
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", a.upstream)
        await asyncio.gather(
            pump(client_reader, up_writer, delay),
            pump(up_reader, client_writer, delay),
            return_exceptions=True,
        )

    server = await asyncio.start_server(handle, "127.0.0.1", a.listen)
    print(
        f"delaying 127.0.0.1:{a.listen} -> :{a.upstream} by {a.delay_ms} ms each way",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
