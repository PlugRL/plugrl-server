"""Read E33's verdicts from each cell's logs and tensorboards.

    python summarise.py

P1 cell by cell, V1, the status rule, then P2, P3 and P4 - PROTOCOL.md's
reading order - and the return at E30's length (iterations 91-100) beside
the end. E31's summarise.py with E33's cells, their own seeds, E30's entropy
check and E28's HalfCheetah rule. Writes `summary.tsv` in E24's columns.
"""

import datetime
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ITERS = 300

# cell: (task, seeds)
CELLS = {
    "fpo-cheetah": ("HalfCheetah-v5", (0, 1, 2)),
    "fpo-hopper": ("Hopper-v5", (3, 4, 5)),
    "fpo-walker": ("Walker2d-v5", (0, 1, 2)),
}
ABSOLUTE = 500.0  # Hopper, Walker2d: mean of iterations 291-300
GAIN = 200.0  # HalfCheetah: the mean of the last ten minus the first
# Every element's entropy depends only on the noise schedule, not the
# weights or the task: E30's `all` logged -0.0040 at every iteration.
ENTROPY, ENTROPY_TOL = -0.004, 0.005


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


def main() -> int:
    rows = {}
    runs_ok = {}
    print("P1  every cell runs end to end")
    for cell, (task, seeds) in CELLS.items():
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
        for seed in seeds:
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
                entropy=curve(run, "losses/entropy"),
            )
        runs_ok[cell] = ok_cell
        minutes = wall_minutes(out)
        print(
            f"    {cell:12s} {task:15s} {'HOLDS' if ok_cell else 'FAILS'}"
            + (f"   ({minutes:.0f} min)" if minutes is not None else "")
        )

    print(
        f"\nV1  logged entropy within {ENTROPY_TOL} of {ENTROPY:+.4f}, every iteration"
    )
    v1 = {}
    for cell, (task, seeds) in CELLS.items():
        values = [v for s in seeds for v in rows[cell, s]["entropy"]]
        v1[cell] = bool(values) and all(abs(v - ENTROPY) <= ENTROPY_TOL for v in values)
        span = f"{min(values):.4f} to {max(values):.4f}" if values else "not logged"
        print(f"    {cell:12s} {span}  {'PASS' if v1[cell] else 'FAIL'}")

    print(
        f"\nstatus  learns: Hopper, Walker2d 291-300 >= {ABSOLUTE:.0f}; "
        f"HalfCheetah 291-300 minus the first >= +{GAIN:.0f}; on 2 of 3"
    )
    status = {}
    for cell, (task, seeds) in CELLS.items():
        passed = sum(passes(task, rows[cell, s]["reward"]) for s in seeds)
        if not (runs_ok[cell] and v1[cell]):
            status[cell] = "not read"
        elif passed >= 2:
            status[cell] = "learns"
        else:
            status[cell] = "did not learn in 1,228,800 steps"
        shown = ", ".join(
            f"{figure(task, rows[cell, s]['reward']):+.1f}"
            if task == "HalfCheetah-v5"
            else f"{figure(task, rows[cell, s]['reward']):.1f}"
            for s in seeds
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

    verdict(
        "P2",
        "fpo-policy learns Hopper under DPPO, on seeds E30 did not use",
        "fpo-hopper",
    )
    verdict("P3", "fpo-policy learns Walker2d under DPPO", "fpo-walker")
    verdict("P4", "fpo-policy learns HalfCheetah under DPPO", "fpo-cheetah")

    print(
        "\nreported: mean return over iterations 91-100 (E30's length), 191-200, 291-300"
    )
    for cell, (task, seeds) in CELLS.items():
        for s in seeds:
            r = rows[cell, s]["reward"]
            spans = [
                mean(r[a:b]) if len(r) >= b else float("nan")
                for a, b in ((90, 100), (190, 200), (290, 300))
            ]
            print(f"    {cell:12s} seed {s}  " + "  ".join(f"{v:8.1f}" for v in spans))

    print(
        "\nreported: first iteration and mean of the last ten, return and episode length"
    )
    lines = [
        "cell\tpolicy\talgorithm\ttask\tclaim\tseed\titerations\truns\tfirst_return\tlast10_return\tfirst_length\tlast10_length"
    ]
    for cell, (task, seeds) in CELLS.items():
        for s in seeds:
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
                        "fpo-policy",
                        "dppo",
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
