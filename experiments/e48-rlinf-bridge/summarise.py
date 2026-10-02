"""E48: judge PROTOCOL.md's checks and predictions from results/.

    python summarise.py

Expects results/<arm>-seed<s>/ for arm local and cross, seeds 0-2, each
with run.txt, rlinf.log and rlinf-logs/ (RLinf's TensorBoard events), as
copied back from guangzhao.
"""

from __future__ import annotations

import pathlib
import re
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R = pathlib.Path(__file__).resolve().parent / "results"
LAPTOP = "100.65.224.34"


def scalars(run: pathlib.Path, tag: str) -> list[float]:
    events = sorted(run.rglob("events.out.tfevents*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    return [e.value for e in acc.Scalars(tag)] if tag in acc.Tags()["scalars"] else []


def main() -> int:
    verdicts = {}
    for arm in ("local", "cross"):
        learned = 0
        times = []
        for seed in (0, 1, 2):
            run = R / f"{arm}-seed{seed}"
            log = (run / "rlinf.log").read_text(encoding="utf-8", errors="replace")
            slots = re.findall(r"plugrl-rlinf: slots (\d+)-(\d+) <- \('([^']+)'", log)
            peers = sorted({p for *_, p in slots})
            expected = "127.0.0.1" if arm == "local" else LAPTOP
            v1 = len(slots) == 16 and peers == [expected]
            steps = scalars(run, "time/step")
            bad = [w for w in ("Traceback", "ClientLost", "TimeoutError") if w in log]
            v2 = len(steps) == 100 and not bad
            returns = scalars(run, "env/return")
            rule = (
                len(returns) >= 6 and statistics.mean(returns[-5:]) >= returns[0] + 200
            )
            learned += rule
            times.append(statistics.median(steps) if steps else float("nan"))
            print(
                f"{arm} seed {seed}: V1 {v1} ({len(slots)} slot lines from {peers}), "
                f"V2 {v2} ({len(steps)} epochs{', ' + ', '.join(bad) if bad else ''}), "
                f"return {returns[0]:.0f} -> last five {statistics.mean(returns[-5:]):.0f} "
                f"({len(returns)} batches): {'learns' if rule else 'does not learn'}"
                if returns
                else f"{arm} seed {seed}: no env/return logged"
            )
            print("   env/return " + " ".join(f"{r:.0f}" for r in returns))
        verdicts[arm] = learned
        print(f"{arm}: learns on {learned} of 3 seeds; median s per epoch "
              + ", ".join(f"{t:.2f}" for t in times))  # fmt: skip
    print(
        f"\nP1 (local learns on >= 2 of 3): {'holds' if verdicts['local'] >= 2 else 'FAILS'}"
    )
    print(
        f"P2 (cross learns on >= 2 of 3): {'holds' if verdicts['cross'] >= 2 else 'FAILS'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
