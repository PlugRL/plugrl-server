"""E57: judge PROTOCOL.md's checks and predictions from results/clean.csv.

    python verdicts.py [results]      # prints, and writes results/verdicts.txt
"""

import csv
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    rows = list(csv.DictReader(open(root / "clean.csv")))
    for r in rows:
        r["seed"], r["harmed"], r["episodes"] = int(r["seed"]), int(r["harmed"]), int(r["episodes"])
        r["clean"] = float(r["clean"])
    out = []

    jobs = (root / "jobs.txt").read_text().split()
    full = sum(r["episodes"] == 50 for r in rows)
    out.append(f"V1: {'holds' if full == len(jobs) == len(rows) else 'FAILS'} - "
               f"{len(rows)} evaluations of {len(jobs)} runs, {full} with 50 episodes")

    by_sha = defaultdict(set)
    for r in rows:
        by_sha[(r["env"], r["sha"])].add(r["clean"])
    shared = [k for k in by_sha if sum(1 for r in rows if (r["env"], r["sha"]) == k) > 1]
    bad = [k for k in shared if len(by_sha[k]) > 1]
    out.append(f"V2: {'holds' if not bad else 'FAILS'} - {len(shared)} weights shared by two or more runs; "
               f"{len(bad)} with differing clean returns {bad[:5]}")

    p1 = {}
    for env in sorted({r["env"] for r in rows}):
        p1[env] = sum(r["harmed"] for r in rows if r["env"] == env and r["fault"] == "none" and r["seed"] >= 10)
    out.append(f"P1: {'holds' if max(p1.values()) <= 1 else 'FALSIFIED'} - band seeds flagged: {p1}")

    p2 = {}
    for env in sorted({r["env"] for r in rows}):
        d = [r for r in rows if r["env"] == env and r["dose"] == "one" and r["kind"] == "boundary" and r["seed"] < 10]
        p2[env] = (sum(r["harmed"] for r in d), len(d))
    ok = all(h <= 0.10 * n for h, n in p2.values())
    out.append(f"P2: {'holds' if ok else 'FALSIFIED'} - harmed at one value (harmed, runs): {p2}")

    d = [r for r in rows if r["env"] == "pendulum" and r["fault"] == "reward:zero" and r["dose"] == "1.0" and r["seed"] < 10]
    ok = len(d) == 3 and all(r["harmed"] for r in d) and all(r["last"] and float(r["last"]) == 0.0 for r in d)
    out.append(f"P3: {'holds' if ok else 'FALSIFIED'} - pendulum reward:zero 1.0: "
               f"{[(r['seed'], r['clean'], r['last'], r['harmed']) for r in d]}")

    h = [r for r in rows if r["env"] == "hopper" and r["fault"] == "obs:f16" and r["dose"] == "1.0" and r["seed"] < 10]
    out.append(f"reported: hopper obs:f16 1.0 (seed, clean, harmed, silent on the curve): "
               f"{[(r['seed'], r['clean'], r['harmed'], r['silent_curve']) for r in h]}")
    text = "\n".join(out)
    (root / "verdicts.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
