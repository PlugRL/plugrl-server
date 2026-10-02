"""E47: what main's and fix's training runs trained, and how long they waited.

    python compare_train.py OUT_DIR

For each arm: the return at every iteration, the SHA-256 of the last
checkpoint's model.safetensors, and the env client's wait per infer.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ARMS = ("main", "fix")


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    rewards, models = {}, {}
    for arm in ARMS:
        run = out / "fpo" / "fpo-policy" / f"halfcheetah-{arm}"
        events = sorted((run / "tensorboard").glob("events.*"))
        acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
        acc.Reload()
        rewards[arm] = [e.value for e in acc.Scalars("rollout/reward")]
        steps = sorted(int(p.name) for p in run.iterdir() if p.name.isdigit())
        data = (run / str(steps[-1]) / "model.safetensors").read_bytes()
        models[arm] = (steps[-1], hashlib.sha256(data).hexdigest())
        summary = out / "runs" / f"e47-{arm}" / "rollout" / "proc_000" / "summary.json"
        t = json.loads(summary.read_text(encoding="utf-8"))
        t = t.get("timing", t)
        print(
            f"{arm}: {len(rewards[arm])} iterations, last checkpoint {models[arm][0]} "
            f"sha256 {models[arm][1][:16]}, infer wait "
            f"{t['infer_wait_s'] / t['infer_calls'] * 1e3:.3f} ms per call, "
            f"collect {t['collect_time_s']:.1f} s"
        )
    print("\niteration  " + "  ".join(f"{arm:>16}" for arm in ARMS))
    for i in range(max(len(r) for r in rewards.values())):
        row = [
            rewards[arm][i] if i < len(rewards[arm]) else float("nan") for arm in ARMS
        ]
        print(f"{i + 1:>9}  " + "  ".join(f"{v:>16.6f}" for v in row))
    print(f"\nmain == fix returns: {rewards['main'] == rewards['fix']}")
    print(f"main == fix model:   {models['main'] == models['fix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
