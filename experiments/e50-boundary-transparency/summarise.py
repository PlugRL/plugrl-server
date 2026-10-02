"""E50: judge PROTOCOL.md's checks and predictions from results/.

    python summarise.py [DIR]      # default: results/

Expects DIR/<env>-<arm>-seed<s>/ with result.json and monitor.csv (and, for
the bridge arm, episodes.csv), plus DIR/<env>-<arm>-seed<s>.log, as run.sh
left them on guangzhao.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ENVS = {
    "pendulum": (102_400, 6_400),
    "halfcheetah": (409_600, 25_600),
}  # steps, vector steps
SEEDS = (0, 1, 2)


def monitor(run: pathlib.Path) -> list[tuple[str, str]]:
    with open(run / "monitor.csv", encoding="utf-8") as f:
        next(f)  # VecMonitor's JSON header
        return [(r["r"], r["l"]) for r in csv.DictReader(f)]


def episode_log(run: pathlib.Path) -> list[tuple[float, int]]:
    with open(run / "episodes.csv", encoding="utf-8", newline="") as f:
        return [(float(r["return"]), int(r["length"])) for r in csv.DictReader(f)]


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    v1 = v2 = p1 = p2 = p3 = p4 = True
    rows = []
    for env, (steps, vector_steps) in ENVS.items():
        for seed in SEEDS:
            a, b = (root / f"{env}-{arm}-seed{seed}" for arm in ("inprocess", "bridge"))
            ra, rb = (json.loads((r / "result.json").read_text()) for r in (a, b))
            complete = ra["steps"] == rb["steps"] == steps and rb.get("client_rc") == 0
            log = (root / f"{env}-bridge-seed{seed}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            slots = log.count("plugrl-bridges: slots 0-15 <-")
            same_weights = ra["weights_sha256"] == rb["weights_sha256"]
            ma, mb = monitor(a), monitor(b)
            same_episodes = ma == mb
            own = episode_log(b)
            agrees = len(own) == len(mb) and all(
                abs(r - float(mr)) < 1e-4 * max(1.0, abs(r)) and n == int(ml)
                for (r, n), (mr, ml) in zip(own, mb)
            )
            extra_ms = (rb["learn_s"] - ra["learn_s"]) / vector_steps * 1000
            v1 &= complete
            v2 &= slots == 1
            p1 &= same_weights
            p2 &= same_episodes
            p3 &= agrees
            p4 &= 0.2 <= extra_ms <= 1.0
            rows.append(
                f"{env:11s} seed {seed}: weights {'same' if same_weights else 'DIFFER'} "
                f"({ra['weights_sha256'][:12]} / {rb['weights_sha256'][:12]}), "
                f"{len(ma)} episodes {'same' if same_episodes else 'DIFFER'}, "
                f"episode log {'agrees' if agrees else 'DISAGREES'}; "
                f"{ra['learn_s']:.1f} s vs {rb['learn_s']:.1f} s "
                f"(+{rb['learn_s'] / ra['learn_s'] - 1:.0%}, {extra_ms:.2f} ms per vector step); "
                f"slots lines {slots}"
            )
    print("\n".join(rows) + "\n")
    for name, ok in (
        ("V1", v1),
        ("V2", v2),
        ("P1", p1),
        ("P2", p2),
        ("P3", p3),
        ("P4", p4),
    ):
        print(f"{name}: {'holds' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
