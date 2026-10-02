"""E47: judge PROTOCOL.md's predictions from results/.

    python summarise.py

Reads results/rtt.tsv, results/idle.tsv and results/train.txt, which is
compare_train.py's output.
"""

from __future__ import annotations

import csv
import pathlib
import re
import statistics

R = pathlib.Path(__file__).resolve().parent / "results"
ARMS = ("main", "fix", "sleep0")


def verdict(ok: bool) -> str:
    return "holds" if ok else "FAILS"


def column(path: pathlib.Path, field: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    return {arm: [float(r[field]) for r in rows if r["arm"] == arm] for arm in ARMS}


def main() -> int:
    rtt = column(R / "rtt.tsv", "ms_per_round_trip")
    idle = column(R / "idle.tsv", "core_fraction")
    print("round trip, ms per infer (5 x 3,000 exchanges)")
    for arm in ARMS:
        print(
            f"  {arm:<7} "
            + " ".join(f"{v:.3f}" for v in rtt[arm])
            + f"   median {statistics.median(rtt[arm]):.3f}"
        )
    print("idle, fraction of a core (5 x 20 s)")
    for arm in ARMS:
        print(
            f"  {arm:<7} "
            + " ".join(f"{v:.4f}" for v in idle[arm])
            + f"   median {statistics.median(idle[arm]):.4f}"
        )

    m = {arm: statistics.median(rtt[arm]) for arm in ARMS}
    i = {arm: statistics.median(idle[arm]) for arm in ARMS}
    complete = all(len(rtt[a]) == 5 and len(idle[a]) == 5 for a in ARMS)
    print(f"\nV1 (Part 1, 2): every run recorded: {complete}")
    print(
        f"P1 fix {m['fix']:.3f} <= 1.2 and main {m['main']:.3f} >= 1.8: {verdict(m['fix'] <= 1.2 and m['main'] >= 1.8)}"
    )
    print(
        f"P2 fix - sleep0 = {m['fix'] - m['sleep0']:+.3f} ms, within 0.2: {verdict(abs(m['fix'] - m['sleep0']) <= 0.2)}"
    )
    print(
        f"P3 fix idle {i['fix']:.4f} <= 0.02 and sleep0 {i['sleep0']:.4f} >= 0.5: {verdict(i['fix'] <= 0.02 and i['sleep0'] >= 0.5)}"
    )
    print(f"   main idle {i['main']:.4f} (reported)")

    text = (R / "train.txt").read_text(encoding="utf-8")
    rc = re.findall(r"(main|fix) (client|server) rc=(\d+)", text)
    print(f"\nV1 (training): exit codes {rc}")
    same_returns = "main == fix returns: True" in text
    same_model = "main == fix model:   True" in text
    print(
        f"P4 same returns {same_returns}, same model {same_model}: {verdict(same_returns and same_model)}"
    )
    waits = dict(re.findall(r"(main|fix): .*?infer wait ([0-9.]+) ms", text))
    collects = dict(re.findall(r"(main|fix): .*?collect ([0-9.]+) s", text))
    w = {k: float(v) for k, v in waits.items()}
    print(
        f"P5 fix {w['fix']:.3f} <= 1.4 and main {w['main']:.3f} >= 1.8: {verdict(w['fix'] <= 1.4 and w['main'] >= 1.8)}"
    )
    c = {k: float(v) for k, v in collects.items()}
    print(
        f"   collect: main {c['main']:.1f} s, fix {c['fix']:.1f} s, ratio {c['main'] / c['fix']:.2f} (reported)"
    )
    sha = re.findall(r"(main|fix): .*?sha256 ([0-9a-f]+)", text)
    print(f"   final weights {sha}; E46's v1a was 6cbdcd2137134eda (reported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
