"""E57: join every clean evaluation with E54's and E56's runs.csv, and count.

    python summarise.py [results]

Writes results/clean.csv (one row per run) and results/summary.txt.
"""

import csv
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
RUNS = [HERE.parent / "e54-boundary-faults/results/runs.csv",
        HERE.parent / "e56-hopper-faults/results/runs.csv"]
CONTROLS = {"none", "delay-1ms", "wire-float64"}
LOUD = {"indices:reorder", "reward:short-array"}


def kind(fault: str) -> str:
    if fault in CONTROLS:
        return "control"
    if fault in LOUD:
        return "loud"
    if fault.startswith("log:"):
        return "log"
    return "boundary"


def load(root: pathlib.Path) -> list[dict]:
    trained = {}
    for path in RUNS:
        for r in csv.DictReader(open(path)):
            trained[(r["env"], r["fault"], r["dose"], int(r["seed"]))] = r
    rows = []
    for f in sorted(root.glob("*/*/*/seed*/clean_eval.json")):
        d = json.loads(f.read_text())
        t = trained[(d["env"], d["fault"], d["dose"], int(d["seed"]))]
        rows.append(dict(
            env=d["env"], fault=d["fault"], dose=d["dose"], seed=int(d["seed"]), kind=kind(d["fault"]),
            episodes=d["episodes"], clean=round(d["mean"], 2), sha=d["weights_sha256"][:8],
            # The loud faults' runs have no `strict` (no return was logged); see AMENDMENT.md.
            rule=int(t["rule"]) if t["rule"] else None, strict=int(t["strict"]) if t["strict"] else None,
            last=float(t["last"]) if t["last"] else None,
        ))
    return rows


def judge(rows: list[dict]) -> dict:
    """Band per env (none, seeds 10-19); harmed = more than 3 s.d. below its mean."""
    bands = {}
    for env in sorted({r["env"] for r in rows}):
        band = [r for r in rows if r["env"] == env and r["fault"] == "none" and r["seed"] >= 10]
        vals = [r["clean"] for r in band]
        bands[env] = dict(mean=st.mean(vals), sd=st.stdev(vals), n=len(vals))
        # Each band seed judged against the other nine.
        for r in band:
            rest = [x["clean"] for x in band if x is not r]
            r["harmed"] = int(r["clean"] < st.mean(rest) - 3 * st.stdev(rest))
    for r in rows:
        if "harmed" not in r:
            b = bands[r["env"]]
            r["harmed"] = int(r["clean"] < b["mean"] - 3 * b["sd"])
        r["silent_curve"] = int(r["rule"] == 0 and r["strict"] == 0)
        r["log_error"] = None if r["last"] is None else round(r["last"] - r["clean"], 2)
    return bands


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    rows = load(root)
    bands = judge(rows)
    cols = ["env", "fault", "dose", "seed", "kind", "episodes", "clean", "harmed", "rule", "strict",
            "silent_curve", "last", "log_error", "sha"]
    with open(root / "clean.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    out = []
    for env, b in bands.items():
        flagged = sum(r["harmed"] for r in rows if r["env"] == env and r["fault"] == "none" and r["seed"] >= 10)
        out.append(f"{env}: band (none, seeds 10-19) mean {b['mean']:.1f} sd {b['sd']:.1f} n {b['n']}; "
                   f"harm bar {b['mean'] - 3 * b['sd']:.1f}; band seeds flagged against the other nine: {flagged} of 10")
    out.append("")
    out.append("boundary-fault runs, seeds 0-2: harmed / runs; harmed among those silent on the curve")
    for env in bands:
        for dose in ("one", "0.001", "0.01", "0.1", "1.0"):
            d = [r for r in rows if r["env"] == env and r["dose"] == dose and r["kind"] == "boundary" and r["seed"] < 10]
            if not d:
                continue
            s = [r for r in d if r["silent_curve"]]
            out.append(f"  {env:12} {dose:6} harmed {sum(r['harmed'] for r in d):3}/{len(d):<3} "
                       f"silent on the curve {len(s):3}, of them harmed {sum(r['harmed'] for r in s):3}")
    out.append("")
    out.append("faults silent on the curve at all three seeds and harmed at two or more (fault, dose: clean returns)")
    for env in bands:
        b = bands[env]
        for (fault, dose) in sorted({(r["fault"], r["dose"]) for r in rows if r["env"] == env and r["kind"] == "boundary"}):
            d = [r for r in rows if r["env"] == env and r["fault"] == fault and r["dose"] == dose and r["seed"] < 10]
            if len(d) == 3 and all(r["silent_curve"] for r in d) and sum(r["harmed"] for r in d) >= 2:
                z = [round((r["clean"] - b["mean"]) / b["sd"], 1) for r in d]
                out.append(f"  {env:12} {fault:20} {dose:6} clean {[r['clean'] for r in d]}  in band s.d. {z}")
    out.append("")
    out.append("log error (logged final return - clean return): min / median / max by kind")
    for env in bands:
        for k in ("control", "boundary", "log"):
            e = [r["log_error"] for r in rows if r["env"] == env and r["kind"] == k and r["log_error"] is not None]
            if e:
                out.append(f"  {env:12} {k:9} n {len(e):3}  {min(e):9.1f} {st.median(e):9.1f} {max(e):9.1f}")
    text = "\n".join(out)
    (root / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
