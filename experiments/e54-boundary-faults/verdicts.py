"""E54: judge PROTOCOL.md's checks and predictions from summarise.py's runs.csv.

    python verdicts.py results      # prints, and writes results/verdicts.txt

Each prediction is judged as PROTOCOL.md words it, on the registered seeds
(0-2), over runs whose fault changed at least one cell.
"""

from __future__ import annotations

import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from faulty_client import LOUD, SILENT  # noqa: E402
from summarise import E50_WEIGHTS  # noqa: E402

LOG = ("log:drop-first", "log:short")
SEEDS = {"0", "1", "2"}


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    with open(root / "runs.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    reg = [r for r in rows if r["seed"] in SEEDS]
    changed = [r for r in reg if r["cells"] not in ("", "0", "None")]
    out = []

    def verdict(name: str, ok: bool, detail: str) -> None:
        out.append(f"{name}: {'holds' if ok else 'FAILS'} - {detail}")

    # V1: every run but the loud faults' completes.
    bad = [r for r in rows if r["fault"] not in LOUD and r["complete"] != "1"]
    verdict(
        "V1",
        not bad,
        f"{len(rows) - len(bad)} of {len(rows)} complete"
        + (f"; not: {bad[:3]}" if bad else ""),
    )
    # V2: none reproduces E50's weights at seeds 0-2.
    v2 = [(r["env"], r["seed"], r["sha"]) for r in reg if r["fault"] == "none"]
    v2_ok = all(
        E50_WEIGHTS[env, int(s)].startswith(sha)
        or sha.startswith(E50_WEIGHTS[env, int(s)])
        for env, s, sha in v2
    )
    verdict("V2", v2_ok and len(v2) == 6, f"{v2}")

    client = [r for r in changed if r["fault"] in SILENT]
    p1_bad = [r for r in client if r["weights"] != "1"]
    verdict(
        "P1",
        not p1_bad,
        f"weights differ in {len(client) - len(p1_bad)} of {len(client)} client-fault runs"
        + (
            f"; not: {[(r['env'], r['fault'], r['dose'], r['seed']) for r in p1_bad]}"
            if p1_bad
            else ""
        ),
    )

    p2 = [
        r
        for r in reg
        if r["fault"] in ("delay-1ms", "wire-float64")
        or (r["fault"] in LOG and r in changed)
    ]
    p2_bad = [r for r in p2 if r["weights"] != "0"]
    verdict(
        "P2",
        not p2_bad,
        f"same weights in {len(p2) - len(p2_bad)} of {len(p2)} control and log-fault runs",
    )

    pend = [r for r in changed if r["env"] == "pendulum" and r["fault"] in SILENT + LOG]
    p3_ok, parts = True, []
    for dose, cap in (
        ("one", 0.10),
        ("0.001", 0.10),
        ("0.01", 0.10),
        ("0.1", 0.10),
        ("1.0", 0.60),
    ):
        rs = [r for r in pend if r["dose"] == dose]
        share = sum(r["rule"] == "1" for r in rs) / len(rs)
        p3_ok &= share <= cap
        parts.append(
            f"{dose}: {sum(r['rule'] == '1' for r in rs)}/{len(rs)} ({share:.1%}, cap {cap:.0%})"
        )
    verdict("P3", p3_ok, "; ".join(parts))

    must = (
        "reward:zero",
        "reward:stale",
        "reward:swap",
        "terminated:spurious",
        "truncated:drop",
    )
    never = tuple(
        f for f in SILENT if f.split(":")[0] in ("obs", "action", "final-obs")
    ) + ("truncated:as-term",)
    p4_miss = [
        r
        for r in changed
        if r["fault"] in must and r["dose"] != "one" and r["record"] != "1"
    ]
    p4_miss += [
        r
        for r in changed
        if r["fault"] == "step:fill"
        and r["dose"] in ("0.001", "0.01", "0.1")
        and r["record"] != "1"
    ]
    p4_miss += [r for r in changed if r["fault"] in LOG and r["record"] != "1"]
    p4_false = [r for r in changed if r["fault"] in never and r["record"] == "1"]
    verdict(
        "P4",
        not p4_miss and not p4_false,
        f"missed {len(p4_miss)} it should catch, caught {len(p4_false)} it should not"
        + (
            f"; missed: {[(r['env'], r['fault'], r['dose'], r['seed']) for r in p4_miss][:8]}"
            if p4_miss
            else ""
        )
        + (
            f"; caught: {[(r['env'], r['fault'], r['dose'], r['seed']) for r in p4_false][:8]}"
            if p4_false
            else ""
        ),
    )

    log_runs = [r for r in changed if r["fault"] in LOG]
    p5_ok = all(r["bridge"] == "1" for r in log_runs) and all(
        r["bridge"] == "0" for r in client
    )
    verdict(
        "P5",
        p5_ok,
        f"bridge caught {sum(r['bridge'] == '1' for r in log_runs)} of {len(log_runs)} log-fault runs and {sum(r['bridge'] == '1' for r in client)} of {len(client)} client-fault runs",
    )

    silent = client + log_runs
    p6_bad = [r for r in silent if r["weights"] != "1" and r["record"] != "1"]
    verdict(
        "P6",
        not p6_bad,
        f"weights or record caught {len(silent) - len(p6_bad)} of {len(silent)}",
    )

    loud = [r for r in reg if r["fault"] in LOUD]
    p7_ok = len(loud) == 6 and all(
        r["raises"] == "1" and float(r["learn_s"]) < 120 for r in loud
    )
    verdict("P7", p7_ok, f"{[(r['fault'], r['seed'], r['learn_s']) for r in loud]}")

    with open(root / "localisation.csv", encoding="utf-8", newline="") as f:
        loc = [r for r in csv.DictReader(f) if r["seed"] in SEEDS]
    p8_bad = [r for r in loc if r["right"] != "1"]
    verdict(
        "P8", not p8_bad, f"right in {len(loc) - len(p8_bad)} of {len(loc)} traced runs"
    )

    cheetah = [
        r for r in changed if r["env"] == "halfcheetah" and r["fault"] in SILENT + LOG
    ]
    p9_ok, parts = True, []
    for dose in ("one", "0.01"):
        rs = [r for r in cheetah if r["dose"] == dose]
        share = sum(r["rule"] == "1" for r in rs) / len(rs)
        p9_ok &= share <= 0.10
        parts.append(
            f"{dose}: {sum(r['rule'] == '1' for r in rs)}/{len(rs)} ({share:.1%})"
        )
    verdict("P9", p9_ok, "; ".join(parts))

    text = "\n".join(out)
    print(text)
    (root / "verdicts.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
