"""E58: judge PROTOCOL.md's checks and predictions from runs.csv, ledger.csv
and each run's clean_eval.json.

    python verdicts.py results      # prints, and writes results/verdicts.txt

Each prediction is judged as PROTOCOL.md words it, on seeds 0-2, over runs
whose fault changed at least one value. E56's verdicts.py, with these changes:
- P1 covers every client fault: Pusher's episodes all end at the time limit,
  where SB3 bootstraps from the final observation, so the final-obs faults
  are read by training;
- P4 drops terminated:drop and terminated:as-trunc, which E58 does not run;
- P8 and P9 judge the clean evaluation, as E57 did.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from faulty_client import CONTROLS, LOUD, SILENT  # noqa: E402

LOG = ("log:drop-first", "log:short")
SEEDS = {"0", "1", "2"}
FINAL_OBS = tuple(f for f in SILENT if f.startswith("final-obs:"))


def read(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    rows = read(root / "runs.csv")
    ledger = read(root / "ledger.csv")
    reg = [r for r in rows if r["seed"] in SEEDS]
    changed = [
        r
        for r in reg
        if r["fault"] not in CONTROLS + LOUD and r["cells"] not in ("", "0", "None")
    ]
    out = []

    def verdict(name: str, ok: bool, detail: str) -> None:
        out.append(f"{name}: {'holds' if ok else 'FAILS'} - {detail}")

    def runs_of(rs) -> list:
        return [(r["fault"], r["dose"], r["seed"]) for r in rs]

    bad = [r for r in rows if r["complete"] != "1"]
    verdict(
        "V1",
        not bad,
        f"{len(rows) - len(bad)} of {len(rows)} complete"
        + (f"; not: {runs_of(bad)[:5]}" if bad else ""),
    )
    none = {r["seed"]: r["sha"] for r in reg if r["fault"] == "none"}
    ctrl = [r for r in reg if r["fault"] in ("delay-1ms", "wire-float64")]
    v2_bad = [r for r in ctrl if r["sha"] != none.get(r["seed"])]
    verdict(
        "V2",
        not v2_bad and len(ctrl) == 6,
        f"controls on none's weights: {len(ctrl) - len(v2_bad)} of {len(ctrl)}",
    )

    client = [r for r in changed if r["fault"] in SILENT]
    p1 = client
    p1_bad = [r for r in p1 if r["weights"] != "1"]
    verdict(
        "P1",
        not p1_bad,
        f"weights differ in {len(p1) - len(p1_bad)} of {len(p1)}"
        + (f"; not: {runs_of(p1_bad)}" if p1_bad else ""),
    )

    logs = [r for r in changed if r["fault"] in LOG]
    p2_bad = [r for r in logs if r["weights"] != "0"]
    verdict(
        "P2",
        not p2_bad,
        f"log faults on none's weights: {len(logs) - len(p2_bad)} of {len(logs)}",
    )

    p3_ok, parts = True, []
    for dose, cap in (("one", 0.10), ("0.01", 0.10), ("1.0", 0.60)):
        rs = [r for r in client + logs if r["dose"] == dose]
        n = sum(r["rule"] == "1" for r in rs)
        p3_ok &= n / len(rs) <= cap
        parts.append(f"{dose}: {n}/{len(rs)} ({n / len(rs):.1%}, cap {cap:.0%})")
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
    miss = [
        r
        for r in changed
        if r["fault"] in must and r["dose"] in ("0.01", "1.0") and r["record"] != "1"
    ]
    miss += [
        r
        for r in changed
        if r["fault"] == "step:fill" and r["dose"] == "0.01" and r["record"] != "1"
    ]
    miss += [r for r in logs if r["record"] != "1"]
    false = [r for r in changed if r["fault"] in never and r["record"] == "1"]
    verdict(
        "P4",
        not miss and not false,
        f"missed {len(miss)}, caught {len(false)} it should not"
        + (f"; missed: {runs_of(miss)[:8]}" if miss else "")
        + (f"; caught: {runs_of(false)[:8]}" if false else ""),
    )

    p5_ok = all(r["bridge"] == "1" for r in logs) and all(
        r["bridge"] == "0" for r in client
    )
    verdict(
        "P5",
        p5_ok,
        f"bridge caught {sum(r['bridge'] == '1' for r in logs)} of {len(logs)} log-fault runs, "
        f"{sum(r['bridge'] == '1' for r in client)} of {len(client)} client-fault runs",
    )

    led = [r for r in ledger if r["seed"] in SEEDS]
    led_client = [
        r for r in led if r["fault"] in SILENT and r["cells"] not in ("", "0")
    ]
    led_quiet = [r for r in led if r["fault"] in CONTROLS + LOG]
    p6_bad = [r for r in led_client if r["caught"] != "1" or r["right"] != "1"]
    p6_false = [r for r in led_quiet if r["caught"] != "0"]
    verdict(
        "P6",
        not p6_bad and not p6_false,
        f"caught and right in {len(led_client) - len(p6_bad)} of {len(led_client)} client-fault runs; "
        f"no difference in {len(led_quiet) - len(p6_false)} of {len(led_quiet)} control and log-fault runs"
        + (
            f"; not: {[(r['fault'], r['dose'], r['seed'], r['found']) for r in p6_bad][:6]}"
            if p6_bad
            else ""
        ),
    )

    caught_ledger = {
        (r["fault"], r["dose"], r["seed"]) for r in led if r["caught"] == "1"
    }
    p7_miss = [r for r in client + logs
               if r["record"] != "1" and (r["fault"], r["dose"], r["seed"]) not in caught_ledger]  # fmt: skip
    verdict(
        "P7",
        not p7_miss,
        f"ledger or record caught {len(client + logs) - len(p7_miss)} of {len(client + logs)}",
    )

    # Clean evaluation (E57's rule): harmed = clean return more than 3 s.d.
    # below the band of fault-free seeds 10-19.
    clean = {}
    for f in root.glob("*/*/*/seed*/clean_eval.json"):
        d = json.loads(f.read_text())
        clean[(d["fault"], d["dose"], str(d["seed"]))] = d["mean"]
    band = [
        clean[("none", "1.0", str(s))]
        for s in range(10, 20)
        if ("none", "1.0", str(s)) in clean
    ]
    flagged = 0
    for i, x in enumerate(band):
        rest = band[:i] + band[i + 1 :]
        flagged += x < statistics.mean(rest) - 3 * statistics.stdev(rest)
    verdict(
        "P8",
        len(band) == 10 and flagged <= 1,
        f"band seeds flagged against the other nine: {flagged} of {len(band)}",
    )
    mu, sd = statistics.mean(band), statistics.stdev(band)
    harmed = {k for k, v in clean.items() if v < mu - 3 * sd}
    low = [r for r in client if r["dose"] in ("one", "0.01")]
    low_h = [r for r in low if (r["fault"], r["dose"], r["seed"]) in harmed]
    verdict(
        "P9",
        len(low_h) <= 0.10 * len(low),
        f"harmed at doses one and 0.01: {len(low_h)} of {len(low)}",
    )
    silent = [r for r in client if r["rule"] == "0" and r["strict"] == "0"]
    silent_h = [r for r in silent if (r["fault"], r["dose"], r["seed"]) in harmed]
    caught = [r for r in client if not (r["rule"] == "0" and r["strict"] == "0")]
    caught_h = [r for r in caught if (r["fault"], r["dose"], r["seed"]) in harmed]
    out.append(
        f"reported: clean band mean {mu:.1f} sd {sd:.1f}, harm bar {mu - 3 * sd:.1f}; "
        f"harmed: {len(silent_h)} of {len(silent)} runs silent on both curve checks, "
        f"{len(caught_h)} of {len(caught)} runs a curve check caught; "
        f"silent and harmed: {runs_of(silent_h)}"
    )
    text = "\n".join(out)
    print(text)
    (root / "verdicts.txt").write_bytes((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
