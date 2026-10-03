"""E54's summarise.py, via E56, with Pusher (E58): what each check says about each fault, at each dose.

E58: Pusher's status rule counts episodes, as Hopper's does: the mean return of
the last tenth of the trainer's log against the first tenth's. Its budget and
bar are fixed by the budget pilot (PUSHER_STEPS, PUSHER_BAR; PROTOCOL.md).

E54's docstring:

    python summarise.py results       # registered: writes runs.csv, by_dose.csv
    python summarise.py pilot

Runs are DIR/<env>/<fault, ':' as '__'>/<dose>/seed<s>/. For each, six checks,
each "catches" the fault or not:
- raises: the run stopped on an error (the bridge rejected a message).
- rule: the trainer's log (monitor.csv) fails the env's status rule, E44's for
  Pendulum and E43's for HalfCheetah. Batches are 16 consecutive episodes.
  Pendulum: batches 20 to the last at least 500 above batches 1-2.
  HalfCheetah: batches 21-25 at least 200 above batch 1.
- strict: the mean of those last batches, in the trainer's log, is more than
  3 standard deviations from the mean of the fault-free runs at seeds 10-19.
- weights: the final weights' SHA-256 differs from the fault-free run's at the
  same seed, on the same machine.
- record: the trainer's log disagrees with the client's record of what the
  environments produced (env_record.csv, in order of the step each episode
  ended on, then env): returns to 1e-4 relative plus 0.01, lengths exactly.
- bridge: the trainer's log disagrees with the bridge's own log
  (episodes.csv), which counts where the rewards arrive, on the trainer's side
  of the wire.
A fault whose run changed no cell (fault.json; for the log faults,
result.json's log_cells) is reported, and left out of the shares caught.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from faulty_client import CONTROLS, LOUD, SILENT  # noqa: E402

LOG = ("log:drop-first", "log:short")
FAULTS = CONTROLS + SILENT + LOG + LOUD
CHECKS = ("raises", "rule", "strict", "weights", "record", "bridge")
STEPS = {"pendulum": 102_400, "halfcheetah": 409_600, "hopper": 1_024_000}
# E50's fault-free bridge runs on guangzhao (seed 9: E50's pilot).
E50_WEIGHTS = {
    ("pendulum", 0): "ec5b8226",
    ("pendulum", 1): "062e92e3",
    ("pendulum", 2): "3a8d968f",
    ("pendulum", 9): "3721266c",
    ("halfcheetah", 0): "8261c03a",
    ("halfcheetah", 1): "86564694",
    ("halfcheetah", 2): "65cc8843",
    ("halfcheetah", 9): "0260ca9d",
}
BAND_SEEDS = range(10, 20)
# E58: fixed by the budget pilot, before the registered runs (PROTOCOL.md).
PUSHER_STEPS = 409_600
PUSHER_BAR = 36.0  # half the budget pilot's mean gain at 409,600 steps (72.6, 73.7)
STEPS["pusher"] = PUSHER_STEPS


def read_csv(path: pathlib.Path, skip_header: bool = False) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        if skip_header:
            next(f)  # VecMonitor's JSON header
        return list(csv.DictReader(f))


def monitor(run: pathlib.Path) -> list[tuple[float, int]]:
    return [(float(r["r"]), int(r["l"])) for r in read_csv(run / "monitor.csv", True)]


def ordered(rows: list[dict], key: str) -> list[tuple[float, int]]:
    rows.sort(key=lambda r: (int(r["step"]), int(r[key])))
    return [(float(r["return"]), int(r["length"])) for r in rows]


def agree(log, truth) -> bool:
    return len(log) == len(truth) and all(
        abs(r - t) <= 1e-4 * max(1.0, abs(t)) + 0.01 and n == m
        for (r, n), (t, m) in zip(log, truth)
    )


def window(env: str, episodes) -> tuple[float, float] | None:
    """(gain, last-window mean) by the env's status rule, or None if too short."""
    returns = [r for r, _ in episodes]
    batches = [
        statistics.mean(returns[i : i + 16]) for i in range(0, len(returns) - 15, 16)
    ]
    if env in ("hopper", "pusher"):
        # Hopper's episodes vary in length, so its windows count episodes: the
        # first and the last tenth of the trainer's log, at least 20 episodes.
        if len(returns) < 20:
            return None
        k = len(returns) // 10
        first, last = statistics.mean(returns[:k]), statistics.mean(returns[-k:])
        return last - first, last
    if env == "pendulum":
        if len(batches) < 20:
            return None
        first, last = statistics.mean(batches[0:2]), statistics.mean(batches[19:])
    else:
        if len(batches) < 25:
            return None
        first, last = batches[0], statistics.mean(batches[20:25])
    return last - first, last


def bar(env: str) -> float:
    if env == "pusher":
        return PUSHER_BAR
    if env == "hopper":
        return 1000.0  # E56's, fixed after its budget pilot (PROTOCOL.md)
    return 500.0 if env == "pendulum" else 200.0


def runs(root: pathlib.Path):
    for res in sorted(root.glob("*/*/*/seed*/result.json")):
        run = res.parent
        env = run.parent.parent.parent.name
        fault = run.parent.parent.name.replace("__", ":")
        yield env, fault, run.parent.name, int(run.name.removeprefix("seed")), run


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    table = {}
    for env, fault, dose, seed, run in runs(root):
        res = json.loads((run / "result.json").read_text())
        info = run / "fault.json"
        cells = json.loads(info.read_text())["cells"] if info.exists() else None
        if fault in LOG:
            cells = res.get("log_cells")
        if fault in CONTROLS:
            cells = 0
        log = monitor(run)
        w = window(env, log)
        table[env, fault, dose, seed] = dict(
            raised=res.get("raised", ""),
            complete=res["steps"] == STEPS[env] and res.get("client_rc") == 0,
            sha=res["weights_sha256"],
            gain=None if w is None else w[0],
            last=None if w is None else w[1],
            record=agree(log, ordered(read_csv(run / "env_record.csv"), "env")),
            bridge=agree(log, ordered(read_csv(run / "episodes.csv"), "slot")),
            cells=cells,
            episodes=len(log),
            learn_s=res["learn_s"],
        )

    rows = []
    for (env, fault, dose, seed), t in sorted(table.items()):
        clean = table.get((env, "none", "1.0", seed))
        band = [table[env, "none", "1.0", s]["last"] for s in BAND_SEEDS
                if (env, "none", "1.0", s) in table]  # fmt: skip
        mu = statistics.mean(band) if len(band) >= 2 else None
        sd = statistics.stdev(band) if len(band) >= 2 else None
        raised = bool(t["raised"])
        caught = dict(
            raises=raised,
            rule=not raised and (t["gain"] is None or t["gain"] < bar(env)),
            strict=None
            if mu is None or raised
            else (t["last"] is None or abs(t["last"] - mu) > 3 * sd),
            weights=None if clean is None or raised else t["sha"] != clean["sha"],
            record=not raised and not t["record"],
            bridge=not raised and not t["bridge"],
        )
        rows.append(
            dict(
                env=env,
                fault=fault,
                dose=dose,
                seed=seed,
                cells=t["cells"],
                **{k: "" if v is None else int(v) for k, v in caught.items()},
                gain="" if t["gain"] is None else round(t["gain"], 1),
                last="" if t["last"] is None else round(t["last"], 1),
                episodes=t["episodes"],
                sha=t["sha"][:8],
                learn_s=t["learn_s"],
                raised=t["raised"][:80],
                complete=int(t["complete"]),
            )
        )
    with open(root / "runs.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)

    # Per fault and dose: how many of the seeds each check caught.
    groups = defaultdict(list)
    for r in rows:
        if r["seed"] in BAND_SEEDS:
            continue
        groups[r["env"], r["fault"], r["dose"]].append(r)
    print(
        f"{'env':11s} {'fault':20s} {'dose':6s} cells  "
        + "  ".join(f"{c:>7s}" for c in CHECKS)
    )
    for (env, fault, dose), rs in sorted(
        groups.items(), key=lambda g: (g[0][0], FAULTS.index(g[0][1]), g[0][2])
    ):
        cells = "/".join(str(r["cells"]) for r in rs)
        counts = []
        for c in CHECKS:
            vals = [r[c] for r in rs if r[c] != ""]
            counts.append(f"{sum(vals)}/{len(vals)}" if vals else "-")
        print(
            f"{env:11s} {fault:20s} {dose:6s} {cells:>5s}  "
            + "  ".join(f"{x:>7s}" for x in counts)
        )

    # The figure: per env, dose and kind of fault, the share of runs with a
    # changed cell that each check caught; "either" is weights or record.
    kinds = {"client": SILENT, "log": LOG, "all": SILENT + LOG}
    shares = []
    for env in ("pendulum", "halfcheetah", "hopper", "pusher"):
        for kind, members in kinds.items():
            for dose in ("one", "0.001", "0.01", "0.1", "1.0"):
                rs = [r for r in rows if r["env"] == env and r["dose"] == dose
                      and r["fault"] in members and r["seed"] not in BAND_SEEDS
                      and r["cells"] not in (0, None)]  # fmt: skip
                if not rs:
                    continue
                line = dict(env=env, kind=kind, dose=dose, runs=len(rs))
                for c in CHECKS:
                    vals = [r[c] for r in rs if r[c] != ""]
                    line[c] = round(sum(vals) / len(vals), 3) if vals else ""
                line["either"] = round(
                    sum(r["weights"] or r["record"] for r in rs) / len(rs), 3
                )
                shares.append(line)
    with open(root / "by_dose.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(
            f, fieldnames=["env", "kind", "dose", "runs", *CHECKS, "either"]
        )
        out.writeheader()
        out.writerows(shares)
    print()
    print(
        "share of runs caught, by dose (runs with a changed cell; either = weights or record)"
    )
    for s in shares:
        print(
            f"  {s['env']:11s} {s['kind']:6s} {s['dose']:6s} n={s['runs']:3d}  "
            + "  ".join(f"{c} {s[c]}" for c in (*CHECKS, "either"))
        )

    print()
    for (env, seed), want in sorted(E50_WEIGHTS.items()):
        t = table.get((env, "none", "1.0", seed))
        if t is not None:
            print(
                f"{env} none at seed {seed}: {t['sha'][:8]}, E50's {want}: "
                f"{'same' if t['sha'].startswith(want) else 'DIFFERENT'}"
            )
    for env in ("pendulum", "halfcheetah", "hopper", "pusher"):
        band = [table[env, "none", "1.0", s]["last"] for s in BAND_SEEDS
                if (env, "none", "1.0", s) in table]  # fmt: skip
        if len(band) >= 2:
            print(
                f"{env} fault-free band, seeds 10-19: mean {statistics.mean(band):.1f}, "
                f"sd {statistics.stdev(band):.1f}, min {min(band):.1f}, max {max(band):.1f}"
            )
        if len(band) >= 3:
            # The strict check's false alarms: each fault-free seed judged by
            # the band of the other nine.
            alarms = 0
            for i, x in enumerate(band):
                rest = band[:i] + band[i + 1 :]
                alarms += abs(x - statistics.mean(rest)) > 3 * statistics.stdev(rest)
            print(
                f"{env} strict check on fault-free seeds 10-19, each against the other "
                f"nine: {alarms} of {len(band)} flagged"
            )
    incomplete = [k for k, t in table.items() if not t["complete"] and not t["raised"]]
    print("incomplete without an error:", incomplete or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
