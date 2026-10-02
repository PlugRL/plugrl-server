"""E52, after the registered runs: what delay_proxy.py itself adds to a round trip.

    python proxy_rtt.py [--delays 0 1 5 25] [--rounds 300]

For each delay D it starts delay_proxy.py in front of a TCP echo server, sends
a 4 KiB message through it and waits for the echo, `--rounds` times, and
reports the median and 90th percentile round trip minus 2D. No bridge, no
trainer, no WebSocket: only the proxy and the loopback.

With --burst 2 each round sends two messages 0.2 ms apart, as E52's client
sends a feedback and then the next infer, and waits for both echoes.
"""

from __future__ import annotations

import argparse
import pathlib
import socket
import statistics
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ECHO, PROXY = 8841, 8842
PAYLOAD = b"x" * 4096


def echo_server(stop: threading.Event) -> None:
    with socket.create_server(("127.0.0.1", ECHO)) as srv:
        srv.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            with conn:
                while data := conn.recv(1 << 16):
                    conn.sendall(data)


def recv_exactly(sock: socket.socket, n: int) -> None:
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("closed")
        got += len(chunk)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delays", type=float, nargs="+", default=[0, 1, 5, 25])
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--burst", type=int, default=1)
    a = p.parse_args()
    stop = threading.Event()
    threading.Thread(target=echo_server, args=(stop,), daemon=True).start()
    time.sleep(0.3)
    for d in a.delays:
        proxy = subprocess.Popen(
            [sys.executable, str(HERE / "delay_proxy.py"), "--listen", str(PROXY),
             "--upstream", str(ECHO), "--delay-ms", str(d)],
            stdout=subprocess.DEVNULL,
        )  # fmt: skip
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", PROXY)) as s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                rtts = []
                for _ in range(a.rounds):
                    start = time.perf_counter()
                    for k in range(a.burst):
                        if k:
                            spin = time.perf_counter() + 0.0002
                            while time.perf_counter() < spin:
                                pass
                        s.sendall(PAYLOAD)
                    recv_exactly(s, a.burst * len(PAYLOAD))
                    rtts.append((time.perf_counter() - start) * 1000)
        finally:
            proxy.terminate()
            proxy.wait()
        excess = sorted(r - 2 * d for r in rtts)
        print(
            f"burst {a.burst}, D {d:5.1f} ms: round trip median {statistics.median(rtts):.3f} ms, "
            f"beyond 2D median {statistics.median(excess):.3f}, "
            f"p90 {excess[int(0.9 * len(excess))]:.3f} ms",
            flush=True,
        )
    stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
