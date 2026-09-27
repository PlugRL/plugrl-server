"""Read E38's verdicts from each cell's logs and tensorboards.

    python summarise.py

P1 cell by cell, V1, the status rule, then P2, P3 and P4 - PROTOCOL.md's
reading order - and the reported returns beside CleanRL's. E33's
summarise.py with E38's cells, 488 iterations, and the learning-rate schedule
as the check that the run was the configuration it claims. Writes
`summary.tsv` in E24's columns.
"""

import datetime
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ITERS = 488
SEEDS = (0, 1, 2)

# cell: (task, CleanRL's reported return for the -v4 task, three seeds)
CELLS = {
    "ppo-cheetah": ("HalfCheetah-v5", "1442.64 +/- 46.03"),
    "ppo-hopper": ("Hopper-v5", "2382.86 +/- 271.74"),
    "ppo-walker": ("Walker2d-v5", "2287.95 +/- 571.78"),
}
ABSOLUTE = 500.0  # Hopper, Walker2d: mean of iterations 479-488
GAIN = 200.0  # HalfCheetah: the mean of the last ten minus the first
LEARNING_RATE = 3e-4


def run_dir(cell: str, seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / cell).glob(f"*/*/{cell}-seed{seed}"))
    return hits[0] if hits else None


def curve(run: pathlib.Path | None, tag: str) -> list[float]:
    if run is None:
        return []
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


def figure(task: str, reward: list[float]) -> float:
    """The number the status rule reads: the gain for HalfCheetah, the level otherwise."""
    if len(reward) < ITERS:
        return float("nan")
    last10 = mean(reward[ITERS - 10 : ITERS])
    return last10 - reward[0] if task == "HalfCheetah-v5" else last10


def passes(task: str, reward: list[float]) -> bool:
    bar = GAIN if task == "HalfCheetah-v5" else ABSOLUTE
    return figure(task, reward) >= bar


def schedule_took(rates: list[float]) -> bool:
    """CleanRL's anneal: iteration i (from 1) learns at (1 - (i-1)/488) * 3e-4."""
    if len(rates) < ITERS:
        return False
    expected = [(1 - i / ITERS) * LEARNING_RATE for i in range(ITERS)]
    return all(abs(r - e) <= 1e-6 * LEARNING_RATE for r, e in zip(rates, expected))


def main() -> int:
    rows = {}
    runs_ok = {}
    print("P1  every cell runs end to end")
    for cell, (task, _) in CELLS.items():
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
            reward = curve(run, "rollout/reward")
            saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
            log_path = RESULTS / cell / f"server-seed{seed}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            # The checkpoint directory's step can be off by one (E16), so
            # saves are counted, not named.
            ok = (
                len(reward) >= ITERS
                and len(saves) >= ITERS // 20
                and bool(log)
                and "Traceback" not in log
                and exits.get(seed) == 0
            )
            ok_cell &= ok
            rows[cell, seed] = dict(
                ok=ok,
                reward=reward,
                length=curve(run, "rollout/length"),
                rate=curve(run, "models/learning_rate"),
            )
        runs_ok[cell] = ok_cell
        minutes = wall_minutes(out)
        print(
            f"    {cell:12s} {task:15s} {'HOLDS' if ok_cell else 'FAILS'}"
            + (f"   ({minutes:.0f} min)" if minutes is not None else "")
        )

    print("\nV1  every iteration learned at CleanRL's annealed rate")
    v1 = {}
    for cell in CELLS:
        v1[cell] = all(schedule_took(rows[cell, s]["rate"]) for s in SEEDS)
        rates = rows[cell, SEEDS[0]]["rate"]
        span = f"{rates[0]:.3g} to {rates[-1]:.3g}" if rates else "not logged"
        print(f"    {cell:12s} {span}  {'PASS' if v1[cell] else 'FAIL'}")

    print(
        f"\nstatus  learns: Hopper, Walker2d 479-488 >= {ABSOLUTE:.0f}; "
        f"HalfCheetah 479-488 minus the first >= +{GAIN:.0f}; on 2 of 3"
    )
    status = {}
    for cell, (task, _) in CELLS.items():
        passed = sum(passes(task, rows[cell, s]["reward"]) for s in SEEDS)
        if not (runs_ok[cell] and v1[cell]):
            status[cell] = "not read"
        elif passed >= 2:
            status[cell] = "learns"
        else:
            status[cell] = "did not learn in 999,424 steps"
        shown = ", ".join(
            f"{figure(task, rows[cell, s]['reward']):+.1f}"
            if task == "HalfCheetah-v5"
            else f"{figure(task, rows[cell, s]['reward']):.1f}"
            for s in SEEDS
        )
        print(f"    {cell:12s} {shown:28s} {passed} of 3  -> {status[cell]}")

    def verdict(name: str, text: str, cell: str) -> None:
        s = status[cell]
        mark = (
            "NOT READ"
            if s == "not read"
            else ("HOLDS" if s == "learns" else "FALSIFIED")
        )
        print(f"\n{name}  {text}\n    {mark}")

    verdict("P2", "gaussian-policy learns Hopper under ppo", "ppo-hopper")
    verdict("P3", "gaussian-policy learns Walker2d under ppo", "ppo-walker")
    verdict("P4", "gaussian-policy learns HalfCheetah under ppo", "ppo-cheetah")

    print(
        "\nreported: mean return over iterations 91-100, 241-250 and 479-488,"
        " beside CleanRL's on the -v4 task"
    )
    for cell, (task, cleanrl) in CELLS.items():
        for s in SEEDS:
            r = rows[cell, s]["reward"]
            spans = [
                mean(r[a:b]) if len(r) >= b else float("nan")
                for a, b in ((90, 100), (240, 250), (478, 488))
            ]
            print(f"    {cell:12s} seed {s}  " + "  ".join(f"{v:8.1f}" for v in spans))
        print(f"    {cell:12s} CleanRL  {cleanrl}")

    print(
        "\nreported: first iteration and mean of the last ten, return and episode length"
    )
    lines = [
        "cell\tpolicy\talgorithm\ttask\tclaim\tseed\titerations\truns\tfirst_return\tlast10_return\tfirst_length\tlast10_length"
    ]
    for cell, (task, _) in CELLS.items():
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
                f"    {cell:12s} seed {s}  return {first:8.1f} -> {last10:8.1f}   "
                f"length {first_len:6.1f} -> {last10_len:6.1f}"
            )
            claim = "learns" if status[cell] == "learns" else "runs"
            lines.append(
                "\t".join(
                    [
                        cell,
                        "gaussian-policy",
                        "ppo",
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
