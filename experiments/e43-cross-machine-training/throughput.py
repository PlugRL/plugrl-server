"""Raw TCP throughput between the two machines, to read the ladder against.

    python throughput.py sink PORT            # on guangzhao: receive and time
    python throughput.py send HOST PORT MIB   # on the laptop: send MIB MiB

The sink prints MiB received, seconds, and MB/s (10^6 bytes), timed from the
first byte to the connection closing.
"""

import socket
import sys
import time


def sink(port: int) -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print("listening", flush=True)
    conn, _ = srv.accept()
    total, start = 0, None
    while True:
        chunk = conn.recv(1 << 20)
        if not chunk:
            break
        if start is None:
            start = time.perf_counter()
        total += len(chunk)
    if start is None:
        print("0.0	0.000	0.0", flush=True)
        return
    secs = time.perf_counter() - start
    print(f"{total / 2**20:.1f}\t{secs:.3f}\t{total / secs / 1e6:.1f}", flush=True)


def send(host: str, port: int, mib: int) -> None:
    s = socket.create_connection((host, port))
    block = bytes(1 << 20)
    for _ in range(mib):
        s.sendall(block)
    s.close()


if __name__ == "__main__":
    if sys.argv[1] == "sink":
        sink(int(sys.argv[2]))
    else:
        send(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
