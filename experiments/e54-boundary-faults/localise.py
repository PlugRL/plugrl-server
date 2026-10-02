"""E54: where a traced run first differs from the fault-free run at its seed.

    python localise.py results            # writes results/localisation.csv

On one machine the fault-free run at a seed is deterministic, so a run with a
fault crosses the same bytes until the fault's first cell, and differs from
there on. Within a vector step the trainer sees, in order: the actions it
sends; then the rewards, episode ends, time limits and final observations of
the feedback; then the next infer's observations. The run is localised to
the first step that differs, the lowest env that differs in it, and the first
of that env's messages that differs.

It is right if that step and env are the fault's first changed cell (the
client's fault.json; the trainer's for the log faults, which change nothing
on the wire), and the message is the one that fault touches:
- obs:*            the next observations, one step earlier (the infer that
                   carried the fault ends the step before, as the trainer
                   counts);
- action:*, step:fill  that step's feedback: the client changed what it did
                   with an action the trainer sent unchanged;
- reward:*         rewards;   final-obs:*  final observations;
- terminated:spurious, truncated:drop  episode ends;
- truncated:as-term  time limits;
- log:*            nothing: the wire is identical to the fault-free run's.
A run whose fault changed no cell, and a control, is right if its wire is
identical. The faults the bridge rejects are not localised: they raise.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

ORDER = ("actions", "rewards", "ends", "truncs", "finals", "nexts")
FEEDBACK = ("rewards", "ends", "truncs", "finals", "nexts")
EXPECTED = {
    "obs": ("nexts",),
    "action": FEEDBACK,
    "step": FEEDBACK,
    "reward": ("rewards",),
    "final-obs": ("finals",),
    "terminated": ("ends",),
    "truncated:drop": ("ends",),
    "truncated:as-term": ("truncs",),
}


def first_difference(a, b) -> tuple[int, str, int] | None:
    """(vector step, message, env) of the first difference, or None."""
    if not np.array_equal(a["first_obs"], b["first_obs"]):
        env = int(np.flatnonzero((a["first_obs"] != b["first_obs"]).any(axis=1))[0])
        return (-1, "nexts", env)
    steps = min(len(a["actions"]), len(b["actions"]))
    # [key, step, env]: where each message differs
    diff = np.stack(
        [
            (a[k][:steps] != b[k][:steps]).reshape(steps, a[k].shape[1], -1).any(axis=2)
            for k in ORDER
        ]
    )
    rows = np.flatnonzero(diff.any(axis=(0, 2)))
    if len(rows):
        t = int(rows[0])
        env = int(np.flatnonzero(diff[:, t].any(axis=0))[0])  # the lowest env first
        key = ORDER[int(np.flatnonzero(diff[:, t, env])[0])]  # then its first message
        return (t, key, env)
    if len(a["actions"]) != len(b["actions"]):
        return (steps, "length", -1)
    return None


def load(path: pathlib.Path) -> dict:
    with np.load(path) as z:  # an NpzFile decompresses on every access
        return {k: z[k] for k in z.files}


def expected(fault: str) -> tuple[str, ...]:
    if fault in EXPECTED:
        return EXPECTED[fault]
    return EXPECTED.get(fault.partition(":")[0], ())


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results")
    rows = []
    for trace in sorted(root.glob("pendulum/*/*/seed*/trace.npz")):
        run = trace.parent
        seed = run.name.removeprefix("seed")
        fault = run.parent.parent.name.replace("__", ":")
        if fault in ("none", "indices:reorder", "reward:short-array"):
            continue  # the reference, and the faults the bridge rejects
        clean = root / "pendulum" / "none" / "1.0" / f"seed{seed}" / "trace.npz"
        where = first_difference(load(trace), load(clean))
        info_path = run / "fault.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        first = info.get("first")
        cells = info.get("cells", 0)
        if fault.startswith("log:"):
            result = json.loads((run / "result.json").read_text())
            cells = result.get("log_cells", 0)
        if (
            cells == 0
            or fault.startswith("log:")
            or fault in ("delay-1ms", "wire-float64")
        ):
            right = where is None  # nothing on the wire changed
        elif where is None or first is None:
            right = False
        else:
            t, key, env = where
            step = first[0] - 1 if fault.startswith("obs:") else first[0]
            right = t == step and env == first[1] and key in expected(fault)
        rows.append(
            dict(
                fault=fault,
                dose=run.parent.name,
                seed=seed,
                cells=cells,
                injected="" if first is None else f"{first[0]}/{first[1]}",
                found="identical"
                if where is None
                else f"{where[0]}/{where[2]} {where[1]}",
                right=int(right),
            )
        )
    out = root / "localisation.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wrong = [r for r in rows if not r["right"]]
    print(
        f"{len(rows)} traced runs with a fault; localised right: {len(rows) - len(wrong)}"
    )
    for r in wrong:
        print("  not right:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
