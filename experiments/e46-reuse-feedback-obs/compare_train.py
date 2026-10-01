"""E46 Part B: what the three runs of train_check.sh trained.

    python compare_train.py OUT_DIR

Prints, for v1a, v1b and v2: the return at every iteration, the server's
reused_observations at its last log, the resync closes in the server log,
and the SHA-256 of the last checkpoint's model.safetensors.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ARMS = ("v1a", "v1b", "v2")


def run_dir(out: pathlib.Path, arm: str) -> pathlib.Path:
    return out / "fpo" / "fpo-policy" / f"halfcheetah-{arm}"


def scalars(run: pathlib.Path, tag: str) -> list[tuple[int, float]]:
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in acc.Scalars(tag)]


def last_model(run: pathlib.Path) -> tuple[int, str]:
    steps = sorted(int(p.name) for p in run.iterdir() if p.name.isdigit())
    if not steps:
        return -1, ""
    data = (run / str(steps[-1]) / "model.safetensors").read_bytes()
    return steps[-1], hashlib.sha256(data).hexdigest()


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    rewards, reused, models = {}, {}, {}
    for arm in ARMS:
        run = run_dir(out, arm)
        rewards[arm] = scalars(run, "rollout/reward")
        counts = scalars(run, "server/reused_observations")
        reused[arm] = counts[-1] if counts else None
        models[arm] = last_model(run)
        log = (out / f"server-{arm}.log").read_text(encoding="utf-8", errors="replace")
        resyncs = log.count("Requesting worker resync")
        print(
            f"{arm}: {len(rewards[arm])} iterations, reused_observations "
            f"{reused[arm]}, resync closes {resyncs}, last checkpoint "
            f"{models[arm][0]} sha256 {models[arm][1][:16]}"
        )
    print("\niteration  " + "  ".join(f"{arm:>16}" for arm in ARMS))
    for i in range(max(len(r) for r in rewards.values())):
        row = [rewards[arm][i][1] if i < len(rewards[arm]) else float("nan") for arm in ARMS]
        print(f"{i + 1:>9}  " + "  ".join(f"{v:>16.6f}" for v in row))
    same_returns = lambda a, b: [v for _, v in rewards[a]] == [v for _, v in rewards[b]]  # noqa: E731
    print(f"\nv1a == v1b returns: {same_returns('v1a', 'v1b')}")
    print(f"v1a == v2  returns: {same_returns('v1a', 'v2')}")
    print(f"v1a == v1b model:   {models['v1a'] == models['v1b']}")
    print(f"v1a == v2  model:   {models['v1a'] == models['v2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
