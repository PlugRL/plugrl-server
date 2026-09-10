"""Run plugrl-server with a chosen scheduler poll interval.

The server's scheduler loop sleeps SCHEDULER_SLEEP_INTERVAL (1 ms by default)
between passes, so an inference request waits for the next pass before it is
even looked at. That wait shows up in every round-trip measurement and has
nothing to do with the network boundary.

Patching the constant and re-measuring separates the two: whatever the
round-trip loses when the interval shrinks was polling latency, not the cost
of crossing a process boundary.

Usage: python server_with_interval.py <seconds> [plugrl-run-server args...]
"""

import sys

import plugrl_server.server.websocket_agent_server as ws_server

interval = float(sys.argv[1])
ws_server.SCHEDULER_SLEEP_INTERVAL = interval
print(f"scheduler poll interval patched to {interval * 1000:.3f} ms", flush=True)

sys.argv = ["plugrl-run-server"] + sys.argv[2:]

from plugrl_server.cli import main  # noqa: E402

raise SystemExit(main())
