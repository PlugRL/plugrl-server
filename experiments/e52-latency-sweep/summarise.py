"""E52: judge PROTOCOL.md's check and predictions from a results directory.

python summarise.py [DIR]      # default: results/
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
E50_INPROCESS = {
    0: "ec5b8226b227f3fa58bd771b411e3673b751ec2c062b010cf0dcef60baed42bc",
    1: "062e92e3bab334bb20d04f963612273166bef926f9edb017a624b133313733d6",
    2: "3a8d968f19e2a8b5dc01540e41c4a64c02cb7a8c99ccdedf4a016008cf8bcd41",
}
ARMS = ["direct", "0", "1", "5", "25"]
VECTOR_STEPS = 6_400


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    v1 = p1 = p2 = True
    for seed, expected in E50_INPROCESS.items():
        results = {}
        for arm in ARMS:
            name = f"pendulum-{arm}-seed{seed}"
            r = json.loads((root / name / "result.json").read_text())
            log = (root / f"{name}.log").read_text(encoding="utf-8", errors="replace")
            v1 &= (
                r["steps"] == 102_400
                and r.get("client_rc") == 0
                and log.count("plugrl-bridges: slots 0-15 <-") == 1
            )
            p1 &= r["weights_sha256"] == expected
            results[arm] = r
        base = results["direct"]["learn_s"]
        line = [f"seed {seed}: direct {base:.1f} s"]
        for arm in ARMS[1:]:
            d = float(arm)
            extra = (results[arm]["learn_s"] - base) / VECTOR_STEPS * 1000
            same = results[arm]["weights_sha256"] == expected
            if d > 0:
                p2 &= 2 * d <= extra <= 2 * d + 3
            line.append(
                f"{arm} ms {results[arm]['learn_s']:.1f} s (+{extra:.2f} ms/step, "
                f"{extra - 2 * d:+.2f} over 2D, weights {'same' if same else 'DIFFER'})"
            )
        print("; ".join(line))
    print()
    for name, ok in (("V1", v1), ("P1", p1), ("P2", p2)):
        print(f"{name}: {'holds' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
