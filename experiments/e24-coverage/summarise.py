"""Read E24's verdicts from each cell's logs and tensorboards.

    python summarise.py

P1 cell by cell, then P2, then the descriptive figures - PROTOCOL.md's
reading order. Writes `summary.tsv`, one row per cell and seed, which is what
the coverage matrix is drawn from.
"""

import datetime
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (0, 1, 2)

# cell: (policy, algorithm, task, iterations, claim)
CELLS = {
    "fpo-walker": ("fpo-policy", "fpo", "Walker2d-v5", 100, "learns"),
    "fpodppo-hopper": ("fpo-policy", "dppo", "Hopper-v5", 20, "runs"),
    "fpodppo-walker": ("fpo-policy", "dppo", "Walker2d-v5", 20, "runs"),
    "dppo-cheetah": ("dppo-policy", "dppo", "HalfCheetah-v5", 20, "runs"),
    "dppo-hopper": ("dppo-policy", "dppo", "Hopper-v5", 20, "runs"),
    "dppo-walker": ("dppo-policy", "dppo", "Walker2d-v5", 20, "runs"),
}


def run_dir(cell: str, seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / cell).glob(f"*/*/{cell}-seed{seed}"))
    return hits[0] if hits else None


def curve(run: pathlib.Path, tag: str) -> list[float]:
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    return [e.value for e in acc.Scalars(tag)]


def wall_minutes(out: str) -> float | None:
    stamps = dict(re.findall(r"^(start|end)\s+(\S+ \S+)$", out, flags=re.M))
    if "start" not in stamps or "end" not in stamps:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    delta = datetime.datetime.strptime(stamps["end"], fmt) - datetime.datetime.strptime(
        stamps["start"], fmt
    )
    return delta.total_seconds() / 60


def main() -> int:
    rows = []
    print("P1  every cell runs end to end")
    runs_ok = {}
    for cell, (policy, algo, task, iters, claim) in CELLS.items():
        out_path = RESULTS / f"{cell}.out"
        out = (
            out_path.read_text(encoding="utf-8", errors="replace")
            if out_path.exists()
            else ""
        )
        exits = {
            int(s): int(rc) for s, rc in re.findall(r"seed (\d) finished rc=(\d+)", out)
        }
        ok_cell = True
        for seed in SEEDS:
            run = run_dir(cell, seed)
            reward = curve(run, "rollout/reward") if run else []
            length = curve(run, "rollout/length") if run else []
            saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
            log_path = RESULTS / cell / f"server-seed{seed}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            traceback = "Traceback" in log
            ok = (
                len(reward) >= iters
                and len(saves) >= iters // 20
                and not traceback
                and exits.get(seed) == 0
            )
            ok_cell &= ok
            last10 = (
                sum(reward[iters - 10 : iters]) / 10
                if len(reward) >= iters
                else float("nan")
            )
            last10_len = (
                sum(length[iters - 10 : iters]) / 10
                if len(length) >= iters
                else float("nan")
            )
            rows.append(
                (
                    cell,
                    policy,
                    algo,
                    task,
                    claim,
                    seed,
                    len(reward),
                    ok,
                    reward[0] if reward else float("nan"),
                    last10,
                    length[0] if length else float("nan"),
                    last10_len,
                )
            )
        runs_ok[cell] = ok_cell
        minutes = wall_minutes(out)
        print(
            f"    {cell:15s} {policy:12s} {algo:5s} {task:15s} "
            f"{'HOLDS' if ok_cell else 'FAILS'}"
            + (f"   ({minutes:.0f} min)" if minutes is not None else "")
        )
        if not ok_cell:
            for r in rows:
                if r[0] == cell and not r[7]:
                    print(f"        seed {r[5]}: {r[6]} of {iters} iterations logged")

    print("\nP2  FPO learns Walker2d: mean of iterations 91-100 >= 500 on 2 of 3")
    if not runs_ok["fpo-walker"]:
        print("    not read - fpo-walker did not run end to end")
    else:
        ends = [r[9] for r in rows if r[0] == "fpo-walker"]
        for seed, end in zip(SEEDS, ends):
            print(f"    seed {seed}: {end:8.1f}")
        passed = sum(e >= 500 for e in ends)
        print(f"    {'HOLDS' if passed >= 2 else 'FALSIFIED'}  ({passed} of 3)")

    print(
        "\nreported: first iteration and mean of the last ten, return and episode length"
    )
    for r in rows:
        print(
            f"    {r[0]:15s} seed {r[5]}  return {r[8]:8.1f} -> {r[9]:8.1f}   "
            f"length {r[10]:6.1f} -> {r[11]:6.1f}"
        )

    header = "cell\tpolicy\talgorithm\ttask\tclaim\tseed\titerations\truns\tfirst_return\tlast10_return\tfirst_length\tlast10_length"
    lines = [header] + [
        "\t".join(
            f"{v:.1f}"
            if isinstance(v, float)
            else str(v).lower()
            if isinstance(v, bool)
            else str(v)
            for v in r
        )
        for r in rows
    ]
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
