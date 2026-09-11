"""Run one cell: start a server, run a client against it, decide what happened.

Most of this file is about not lying. Earlier hand-written harnesses in this
project reported a throughput of 118879 exchanges/s from a client that had
died on connect, and reported a configuration as broken when the real problem
was a missing system package. So: a run is only OK if the client's own
summary says it finished the episodes it was asked for, the server is checked
for having started at all, and "could not attempt" is a separate outcome from
"attempted and failed".
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from plugrl_sweep.cell import Cell
from plugrl_sweep.outcome import CellResult, Outcome, classify

SERVER_READY_MARKER = "listening"


@dataclass
class Executables:
    server: str
    client: str


def free_port(attempts: int = 20) -> int:
    """Ask the OS for a port nobody is using.

    There is an unavoidable gap between closing this socket and the server
    binding it, so callers should be ready to retry rather than assume.
    """
    for _ in range(attempts):
        with contextlib.closing(socket.socket()) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port > 1024:
            return port
    raise RuntimeError("could not find a free port")


def wait_for_server(log_path: Path, process: subprocess.Popen, timeout: float) -> bool:
    """True once the server says it is listening, False if it died or hung."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if SERVER_READY_MARKER in text.lower():
                return True
        time.sleep(0.1)
    return False


def find_client_summary(run_dir: Path) -> dict | None:
    """The client writes summary.json somewhere under its output directory."""
    candidates = sorted(run_dir.rglob("summary.json"))
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def run_cell(
    cell: Cell,
    *,
    executables: Executables,
    output_dir: Path,
    host: str = "127.0.0.1",
    server_start_timeout: float = 120.0,
    client_timeout: float = 600.0,
) -> CellResult:
    run_dir = Path(output_dir) / cell.key
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log = run_dir / "server.log"
    client_log = run_dir / "client.log"

    def failure(detail: str, outcome: Outcome = Outcome.FAILED) -> CellResult:
        return CellResult(
            key=cell.key,
            env_uid=cell.env_uid,
            policy_uid=cell.policy_uid,
            algo_uid=cell.algo_uid,
            seed=cell.seed,
            outcome=outcome,
            detail=detail,
        )

    port = free_port()
    started_at = time.monotonic()

    with server_log.open("w", encoding="utf-8") as server_out:
        server = subprocess.Popen(
            cell.server_command(executables.server, port),
            stdout=server_out,
            stderr=subprocess.STDOUT,
            cwd=run_dir,
        )
        try:
            if not wait_for_server(server_log, server, server_start_timeout):
                tail = _tail(server_log)
                # A server that cannot start says nothing about the env, but it
                # does say something about the policy or algorithm it was given.
                return failure(f"server did not start: {tail}")

            timed_out = False
            try:
                completed = subprocess.run(
                    cell.client_command(executables.client, port, host),
                    stdout=client_log.open("w", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                    cwd=run_dir,
                    timeout=client_timeout,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)

    duration = time.monotonic() - started_at
    stdout = client_log.read_text(encoding="utf-8", errors="replace")
    summary = find_client_summary(run_dir)
    episodes = summary.get("completed_episodes") if summary else None

    outcome = classify(
        returncode=returncode,
        stdout=stdout,
        episodes_expected=cell.num_episodes,
        episodes_completed=episodes,
        env_uid=cell.env_uid,
        timed_out=timed_out,
    )

    return CellResult(
        key=cell.key,
        env_uid=cell.env_uid,
        policy_uid=cell.policy_uid,
        algo_uid=cell.algo_uid,
        seed=cell.seed,
        outcome=outcome,
        duration_s=round(duration, 2),
        episodes_completed=episodes,
        detail="" if outcome is Outcome.OK else _tail(client_log),
        timing=(summary or {}).get("timing"),
    )


def _tail(path: Path, lines: int = 3, width: int = 300) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    interesting = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("|")
    ]
    return " / ".join(interesting[-lines:])[:width]
