"""E54: merge each run's result.json, fault.json and log tails into one file.

    python merge_runs.py results      # writes results/runs.jsonl

One line per run: its env, fault, dose and seed (from its directory), its
result.json and fault.json, and the last line of its client.log and
train.log.
"""

from __future__ import annotations

import json
import pathlib
import sys


def last_line(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results")
    out = []
    for res in sorted(root.glob("*/*/*/seed*/result.json")):
        run = res.parent
        line = {
            "env": run.parent.parent.parent.name,
            "fault": run.parent.parent.name.replace("__", ":"),
            "dose": run.parent.name,
            "seed": int(run.name.removeprefix("seed")),
            "result": json.loads(res.read_text()),
        }
        if (run / "fault.json").exists():
            line["fault_json"] = json.loads((run / "fault.json").read_text())
        line["client_log"] = last_line(run / "client.log")
        line["train_log"] = last_line(run / "train.log")
        out.append(json.dumps(line, sort_keys=True))
    (root / "runs.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(len(out), "runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
