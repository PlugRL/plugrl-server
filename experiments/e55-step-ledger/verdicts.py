"""E55: judge PROTOCOL.md's checks and predictions.

    python verdicts.py results      # prints, and writes results/verdicts.txt

Reads, for the same-machine runs, results/ledger.csv (ledger.py) and
results/runs.csv (summarise.py); for the runs across machines, the same two
files under results/cross/.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

CONTROLS = ("none", "delay-1ms", "wire-float64")
LOG = ("log:drop-first", "log:short")
E50 = {"0": "ec5b8226", "1": "062e92e3", "2": "3a8d968f"}


def read(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results")
    same_l, same_r = read(root / "ledger.csv"), read(root / "runs.csv")
    cross_l, cross_r = (
        read(root / "cross" / "ledger.csv"),
        read(root / "cross" / "runs.csv"),
    )
    out = []

    def verdict(name: str, ok: bool, detail: str) -> None:
        out.append(f"{name}: {'holds' if ok else 'FAILS'} - {detail}")

    def client(rows):
        return [
            r
            for r in rows
            if r["fault"] not in CONTROLS + LOG and r["cells"] not in ("", "0")
        ]

    # V1: every run completes. Same machine: runs.csv's complete; across
    # machines the client's exit is in its client.log.
    v1_bad = [r for r in same_r if r["complete"] != "1"]
    cross_bad = []
    for r in cross_r:
        run = (
            root
            / "cross"
            / r["env"]
            / r["fault"].replace(":", "__")
            / r["dose"]
            / f"seed{r['seed']}"
        )
        log = (run / "client.log").read_text(encoding="utf-8", errors="replace")
        res = json.loads((run / "result.json").read_text())
        if "client exit 0" not in log or res["steps"] != 102_400:
            cross_bad.append(r["fault"])
    verdict(
        "V1",
        not v1_bad and not cross_bad,
        f"{len(same_r) - len(v1_bad)} of {len(same_r)} same-machine and "
        f"{len(cross_r) - len(cross_bad)} of {len(cross_r)} cross-machine runs complete",
    )
    none = {r["seed"]: r["sha"] for r in same_r if r["fault"] == "none"}
    verdict(
        "V2",
        all(none.get(s, "").startswith(h) for s, h in E50.items()),
        f"none: {none}",
    )

    c = client(same_l)
    p1_bad = [r for r in c if r["caught"] != "1" or r["right"] != "1"]
    verdict(
        "P1",
        not p1_bad,
        f"caught and right in {len(c) - len(p1_bad)} of {len(c)} client-fault runs"
        + (
            f"; not: {[(r['fault'], r['dose'], r['seed'], r['found']) for r in p1_bad][:6]}"
            if p1_bad
            else ""
        ),
    )

    quiet = [r for r in same_l + cross_l if r["fault"] in CONTROLS + LOG]
    p2_bad = [r for r in quiet if r["caught"] != "0"]
    verdict(
        "P2",
        not p2_bad,
        f"no difference in {len(quiet) - len(p2_bad)} of {len(quiet)} control and log-fault runs",
    )

    cc = client(cross_l)
    cross_faults = {r["fault"] for r in cc}
    p3_bad = [r for r in cc if r["caught"] != "1" or r["right"] != "1"]
    cross_none = [r for r in cross_l if r["fault"] == "none"]
    p3_ok = (
        not p3_bad
        and len(cross_faults) == 21
        and all(r["caught"] == "0" for r in cross_none)
        and cross_none
    )
    verdict(
        "P3",
        bool(p3_ok),
        f"caught and right in {len(cc) - len(p3_bad)} of {len(cc)} cross-machine client-fault runs "
        f"({len(cross_faults)} faults); none differs: {[r['caught'] for r in cross_none]}",
    )

    cross_none_sha = [r["sha"] for r in cross_r if r["fault"] == "none"]
    verdict(
        "P4",
        bool(cross_none_sha) and not cross_none_sha[0].startswith(E50["0"]),
        f"across machines none ended on {cross_none_sha}, guangzhao's on {E50['0']}",
    )

    def caught_either(ledger_rows, run_rows):
        record = {(r["fault"], r["dose"], r["seed"]): r["record"] for r in run_rows}
        rows = [
            r
            for r in ledger_rows
            if r["fault"] not in CONTROLS and r["cells"] not in ("", "0")
        ]
        miss = [
            r
            for r in rows
            if r["caught"] != "1"
            and record.get((r["fault"], r["dose"], r["seed"])) != "1"
        ]
        return rows, miss

    rows_s, miss_s = caught_either(same_l, same_r)
    rows_c, miss_c = caught_either(cross_l, cross_r)
    verdict(
        "P5",
        not miss_s and not miss_c,
        f"ledger or record caught {len(rows_s) - len(miss_s)} of {len(rows_s)} same-machine and "
        f"{len(rows_c) - len(miss_c)} of {len(rows_c)} cross-machine runs with a changed value",
    )

    text = "\n".join(out)
    print(text)
    (root / "verdicts.txt").write_bytes((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
