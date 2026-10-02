"""E55: the trainer's view of each step against the environment's own ledger.

    python ledger.py results            # writes results/ledger.csv

For each run with both trace.npz (what the trainer sent and received, step by
step, from train.py --trace) and ledger.npz (what each environment took and
gave, from faulty_client.py --ledger), and for each env, the two step
sequences are compared in order, value for value:
- obs: the observation the trainer acted on, against the one the environment
  had given before that step;
- action: the action the trainer sent, against the one the environment was
  handed;
- reward: the reward the trainer got, against the one the environment
  returned;
- end: whether the trainer saw the episode end, against terminated or
  truncated;
- limit: whether the trainer saw a time limit (SB3's TimeLimit.truncated),
  against truncated and not terminated;
- final: at an episode end, the observation the trainer got as the final one,
  against the one the environment returned;
- steps: the number of steps on each side.
No fault-free run is needed, and the two sides need not be on one machine.

A run is caught if anything differs. The first difference is (step, env,
field), the lowest step first, then the lowest env, then the order above. It
is right if that step and env are the fault's first changed cell
(fault.json), and the field is the one the fault touches.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

FIELDS = ("obs", "action", "reward", "end", "limit", "final", "steps")
EXPECTED = {
    "obs": ("obs",),
    "action": ("action",),
    "reward": ("reward",),
    "final-obs": ("final",),
    "terminated:spurious": ("end", "limit"),
    "truncated:drop": ("end",),
    "truncated:as-term": ("limit",),
    "step:fill": FIELDS,
}


def load(path: pathlib.Path) -> dict:
    with np.load(path) as z:  # an NpzFile decompresses on every access
        return {k: z[k] for k in z.files}


def differences(trace: dict, ledger: dict) -> tuple[int, tuple[int, int, str] | None]:
    """(cells that differ, first difference as (step, env, field))."""
    steps, n = trace["rewards"].shape
    acted = np.concatenate([trace["first_obs"][None], trace["nexts"][:-1]])
    count, first = 0, None
    for i in range(n):
        k = len(ledger[f"reward{i}"])
        m = min(k, steps)
        if m == 0:  # one side has no steps at all (every step skipped, say)
            count += abs(k - steps)
            if k != steps and (first is None or (0, i) < (first[0], first[1])):
                first = (0, i, "steps")
            continue
        te, tr = ledger[f"terminated{i}"][:m], ledger[f"truncated{i}"][:m]
        ends = trace["ends"][:m, i]
        per_field = {
            "obs": (acted[:m, i] != ledger[f"obs_in{i}"][:m])
            .reshape(m, -1)
            .any(axis=1),
            "action": (trace["actions"][:m, i] != ledger[f"action{i}"][:m])
            .reshape(m, -1)
            .any(axis=1),
            "reward": trace["rewards"][:m, i] != ledger[f"reward{i}"][:m],
            "end": ends != (te | tr),
            "limit": trace["truncs"][:m, i] != (tr & ~te),
            "final": ends
            & (trace["finals"][:m, i] != ledger[f"obs_out{i}"][:m])
            .reshape(m, -1)
            .any(axis=1),
        }
        bad = np.zeros(m, dtype=np.bool_)
        for f in FIELDS[:-1]:
            bad |= per_field[f]
        count += int(bad.sum()) + abs(k - steps)
        rows = np.flatnonzero(bad)
        if len(rows):
            t = int(rows[0])
            field = next(f for f in FIELDS[:-1] if per_field[f][t])
            here = (t, i, field)
        elif k != steps:
            here = (m, i, "steps")
        else:
            continue
        if first is None or (here[0], here[1]) < (first[0], first[1]):
            first = here
    return count, first


def expected(fault: str) -> tuple[str, ...]:
    return EXPECTED.get(fault, EXPECTED.get(fault.partition(":")[0], ()))


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results")
    rows = []
    for trace_path in sorted(root.glob("**/seed*/trace.npz")):
        run = trace_path.parent
        ledger_path = run / "ledger.npz"
        if not ledger_path.exists():
            continue
        fault = run.parent.parent.name.replace("__", ":")
        info = (
            json.loads((run / "fault.json").read_text())
            if (run / "fault.json").exists()
            else {}
        )
        result = json.loads((run / "result.json").read_text())
        cells = (
            result.get("log_cells")
            if fault.startswith("log:")
            else info.get("cells", 0)
        )
        count, first = differences(load(trace_path), load(ledger_path))
        injected = info.get("first")
        if (
            fault.startswith("log:")
            or fault in ("none", "delay-1ms", "wire-float64")
            or not cells
        ):
            right = first is None  # nothing that crossed should differ
        else:
            right = (
                first is not None
                and injected is not None
                and (first[0], first[1]) == (injected[0], injected[1])
                and first[2] in expected(fault)
            )
        rows.append(
            dict(
                run=str(run.relative_to(root)).replace("\\", "/"),
                fault=fault,
                dose=run.parent.name,
                seed=run.name.removeprefix("seed"),
                cells=cells,
                caught=int(first is not None),
                differing=count,
                injected="" if injected is None else f"{injected[0]}/{injected[1]}",
                found="" if first is None else f"{first[0]}/{first[1]} {first[2]}",
                right=int(right),
            )
        )
    with open(root / "ledger.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wrong = [r for r in rows if not r["right"]]
    print(f"{len(rows)} runs compared; right: {len(rows) - len(wrong)}")
    for r in wrong:
        print("  not right:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
