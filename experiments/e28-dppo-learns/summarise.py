"""Read E28's verdicts from each cell's logs and tensorboards.

    python summarise.py

P1 cell by cell, then the status rule applied to every cell, then P2, P3,
P4 - PROTOCOL.md's reading order. Writes `summary.tsv`, one row per cell and
seed, in E24's columns with the cell's status as its claim, so the coverage
matrix reads E24, E27 and E28 the same way.
"""

import datetime
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (0, 1, 2)
ITERS = 100

# cell: (policy, algorithm, task)
CELLS = {
    "dppo-hopper": ("dppo-policy", "dppo", "Hopper-v5"),
    "dppo-walker": ("dppo-policy", "dppo", "Walker2d-v5"),
    "dppo-cheetah": ("dppo-policy", "dppo", "HalfCheetah-v5"),
    "fpodppo-hopper": ("fpo-policy", "dppo", "Hopper-v5"),
    "fpodppo-walker": ("fpo-policy", "dppo", "Walker2d-v5"),
}
ABSOLUTE = 500.0  # Hopper, Walker2d: mean of iterations 91-100
GAIN = 200.0  # HalfCheetah: mean of iterations 91-100 minus the first


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


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def seed_passes(task: str, reward: list[float]) -> bool:
    if len(reward) < ITERS:
        return False
    last10 = mean(reward[ITERS - 10 : ITERS])
    if task == "HalfCheetah-v5":
        return last10 - reward[0] >= GAIN
    return last10 >= ABSOLUTE


def main() -> int:
    rows = {}
    print("P1  every cell runs end to end")
    runs_ok = {}
    for cell, (policy, algo, task) in CELLS.items():
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
            saves = (
                [int(p.name) for p in run.iterdir() if p.name.isdigit()] if run else []
            )
            log_path = RESULTS / cell / f"server-seed{seed}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            # The checkpoint directory's step can be off by one (E16), so
            # the last save is found by count, not by name.
            ok = (
                len(reward) >= ITERS
                and len(saves) >= ITERS // 20
                and bool(log)
                and "Traceback" not in log
                and exits.get(seed) == 0
            )
            ok_cell &= ok
            rows[cell, seed] = dict(
                ok=ok, reward=reward, length=length, passes=seed_passes(task, reward)
            )
        runs_ok[cell] = ok_cell
        minutes = wall_minutes(out)
        print(
            f"    {cell:15s} {policy:12s} {task:15s} "
            f"{'HOLDS' if ok_cell else 'FAILS'}"
            + (f"   ({minutes:.0f} min)" if minutes is not None else "")
        )
        for seed in SEEDS:
            r = rows[cell, seed]
            if not r["ok"]:
                print(
                    f"        seed {seed}: {len(r['reward'])} of {ITERS} iterations logged"
                )

    print(
        f"\nstatus  learns: mean of iterations 91-100 >= {ABSOLUTE:.0f} "
        f"(HalfCheetah: >= first + {GAIN:.0f}) on 2 of 3"
    )
    status = {}
    for cell, (policy, algo, task) in CELLS.items():
        passed = sum(rows[cell, s]["passes"] for s in SEEDS)
        if not runs_ok[cell]:
            status[cell] = "did not run"
        elif passed >= 2:
            status[cell] = "learns"
        else:
            status[cell] = "did not learn in 409,600 steps"
        figures = []
        for s in SEEDS:
            reward = rows[cell, s]["reward"]
            last10 = (
                mean(reward[ITERS - 10 : ITERS])
                if len(reward) >= ITERS
                else float("nan")
            )
            if task == "HalfCheetah-v5" and reward:
                figures.append(f"{last10 - reward[0]:+.1f}")
            else:
                figures.append(f"{last10:.1f}")
        print(
            f"    {cell:15s} {', '.join(figures):28s} {passed} of 3  -> {status[cell]}"
        )

    def verdict(name: str, text: str, holds: bool | None) -> None:
        mark = (
            "not read - a cell did not run"
            if holds is None
            else ("HOLDS" if holds else "FALSIFIED")
        )
        print(f"\n{name}  {text}\n    {mark}")

    def learns(cell: str) -> bool | None:
        return None if status[cell] == "did not run" else status[cell] == "learns"

    verdict("P2", "dppo-policy learns Walker2d", learns("dppo-walker"))
    verdict("P3", "dppo-policy learns Hopper", learns("dppo-hopper"))
    h, w = learns("fpodppo-hopper"), learns("fpodppo-walker")
    verdict(
        "P4",
        "fpo-policy under DPPO learns neither Hopper nor Walker2d",
        None if h is None or w is None else not h and not w,
    )

    print(
        "\nreported: first iteration and mean of the last ten, return and episode length"
    )
    lines = [
        "cell\tpolicy\talgorithm\ttask\tclaim\tseed\titerations\truns\tfirst_return\tlast10_return\tfirst_length\tlast10_length"
    ]
    for cell, (policy, algo, task) in CELLS.items():
        for s in SEEDS:
            r = rows[cell, s]
            reward, length = r["reward"], r["length"]
            first = reward[0] if reward else float("nan")
            last10 = (
                mean(reward[ITERS - 10 : ITERS])
                if len(reward) >= ITERS
                else float("nan")
            )
            first_len = length[0] if length else float("nan")
            last10_len = (
                mean(length[ITERS - 10 : ITERS])
                if len(length) >= ITERS
                else float("nan")
            )
            print(
                f"    {cell:15s} seed {s}  return {first:8.1f} -> {last10:8.1f}   "
                f"length {first_len:6.1f} -> {last10_len:6.1f}"
            )
            claim = "learns" if status[cell] == "learns" else "runs"
            lines.append(
                "\t".join(
                    [
                        cell,
                        policy,
                        algo,
                        task,
                        claim,
                        str(s),
                        str(len(reward)),
                        str(r["ok"]).lower(),
                        f"{first:.1f}",
                        f"{last10:.1f}",
                        f"{first_len:.1f}",
                        f"{last10_len:.1f}",
                    ]
                )
            )
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
