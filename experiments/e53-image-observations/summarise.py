"""E53: judge PROTOCOL.md's checks and predictions from a results directory.

    python summarise.py [DIR] [LOW_MS HIGH_MS]   # default: results/, P3's band

Expects DIR/pong-<arm>-seed<s>/ with result.json and monitor.csv (and, for the
bridge arm, episodes.csv), plus DIR/pong-bridge-seed<s>.log.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
VECTOR_STEPS = 40_960 // 16


def monitor(run: pathlib.Path) -> list[tuple[str, str]]:
    with open(run / "monitor.csv", encoding="utf-8") as f:
        next(f)  # VecMonitor's JSON header
        return [(r["r"], r["l"]) for r in csv.DictReader(f)]


def episode_log(run: pathlib.Path) -> list[tuple[float, int]]:
    with open(run / "episodes.csv", encoding="utf-8", newline="") as f:
        return [(float(r["return"]), int(r["length"])) for r in csv.DictReader(f)]


def main() -> int:
    args = sys.argv[1:]
    root = pathlib.Path(args[0]) if args else HERE / "results"
    low, high = (float(args[1]), float(args[2])) if len(args) == 3 else (0.0, 1e9)
    v1 = p1 = p2 = p3 = True
    for seed in SEEDS:
        a, b = (root / f"pong-{arm}-seed{seed}" for arm in ("inprocess", "bridge"))
        ra, rb = (json.loads((r / "result.json").read_text()) for r in (a, b))
        log = (root / f"pong-bridge-seed{seed}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        v1 &= (
            ra["steps"] == rb["steps"] == 40_960
            and rb.get("client_rc") == 0
            and log.count("plugrl-bridges: slots 0-15 <-") == 1
            and "uint8" in rb["observation_space"]
        )
        same = ra["weights_sha256"] == rb["weights_sha256"]
        ma, mb = monitor(a), monitor(b)
        own = episode_log(b)
        agrees = len(own) == len(mb) and all(
            abs(r - float(mr)) < 1e-4 * max(1.0, abs(r)) and n == int(ml)
            for (r, n), (mr, ml) in zip(own, mb)
        )
        extra = (rb["learn_s"] - ra["learn_s"]) / VECTOR_STEPS * 1000
        p1 &= same
        p2 &= ma == mb and agrees
        p3 &= low <= extra <= high
        print(
            f"seed {seed}: weights {'same' if same else 'DIFFER'} "
            f"({ra['weights_sha256'][:12]} / {rb['weights_sha256'][:12]}), "
            f"{len(ma)} episodes {'same' if ma == mb else 'DIFFER'}, "
            f"episode log {'agrees' if agrees else 'DISAGREES'}; "
            f"{ra['learn_s']:.1f} s vs {rb['learn_s']:.1f} s "
            f"(+{rb['learn_s'] / ra['learn_s'] - 1:.0%}, {extra:.2f} ms per vector step)"
        )
    print()
    for name, ok in (("V1", v1), ("P1", p1), ("P2", p2), ("P3", p3)):
        print(f"{name}: {'holds' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
