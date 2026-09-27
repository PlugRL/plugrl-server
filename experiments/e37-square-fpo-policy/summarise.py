"""Read E37's verdicts from each cell's logs and tensorboards.

    python summarise.py [results-dir]

P1, V1, the status rule, then P2 and P3 - PROTOCOL.md's reading order - and
the success by window. A cell's start is the mean `rollout/success` of the
iterations collected by the cloned policy unchanged (FPO: 1-2, DPPO: 1-3,
after their critic-only iterations); its end is the mean of the last ten.
Writes `summary.tsv`.
"""

import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
SEEDS = (0, 1, 2)
GAIN = 0.2  # learns: end minus start, in success rate, on 2 of 3

# cell: (algorithm directory, iterations, iterations before the first update,
#        what its config line must show)
CELLS = {
    "fpo-square": (
        "fpo",
        60,
        2,
        (
            "output_mode='u'",
            "cfm_loss_steps=4",
            "cfm_loss_dims=7",
            "cfm_loss_sum_over_steps=True",
            "ratio_per_sample=True",
            "n_critic_warmup_itrs=1",
            "clipping_epsilon=0.01",
            "restore='except-critic'",
        ),
    ),
    "dppo-square": (
        "dppo",
        40,
        3,
        (
            "DPPOAlgoConfigSquare(",
            "sampling_noise_level=1.0",
            "logprob_noise_level=1.0",
            "batch_size=500",
            "restore='except-critic'",
        ),
    ),
}


def run_dir(cell: str, algo: str, seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / cell).glob(f"{algo}/fpo-policy/{cell}-seed{seed}"))
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


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    rows, runs_ok, v1 = {}, {}, {}
    print("P1  every cell runs end to end")
    for cell, (algo, iters, start, _) in CELLS.items():
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
            run = run_dir(cell, algo, seed)
            log_path = RESULTS / cell / f"server-seed{seed}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            success = curve(run, "rollout/success")
            saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
            ok = (
                len(success) >= iters
                and len(saves) >= iters // 10
                and bool(log)
                and "Traceback" not in log
                and exits.get(seed) == 0
            )
            ok_cell &= ok
            rows[cell, seed] = dict(
                log=log, success=success, reward=curve(run, "rollout/reward")
            )
        runs_ok[cell] = ok_cell
        print(f"    {cell:12s} {'HOLDS' if ok_cell else 'FAILS'}")

    print("\nV1  every server started from the clone with its cell's settings")
    for cell, (_, _, _, wanted) in CELLS.items():
        missing = set()
        for seed in SEEDS:
            log = rows[cell, seed]["log"]
            if "Restoring from" not in log or "with restore=except-critic" not in log:
                missing.add("the restore line")
            missing |= {w for w in wanted if w not in log}
        v1[cell] = not missing
        print(
            f"    {cell:12s} {'PASS' if v1[cell] else f'FAIL - missing {sorted(missing)}'}"
        )

    print(
        f"\nstatus  learns: success, mean of the last ten minus the start, at least +{GAIN:.1f} on 2 of 3"
    )
    status = {}
    for cell, (_, iters, start, _) in CELLS.items():
        gains = []
        for seed in SEEDS:
            s = rows[cell, seed]["success"]
            a = mean(s[:start])
            b = mean(s[iters - 10 : iters]) if len(s) >= iters else float("nan")
            gains.append(b - a)
            print(f"    {cell:12s} seed {seed}  {a:.3f} -> {b:.3f}  ({b - a:+.3f})")
        passed = sum(g >= GAIN for g in gains)
        readable = runs_ok[cell] and v1[cell]
        status[cell] = (
            "not read"
            if not readable
            else ("learns" if passed >= 2 else "does not learn")
        )
        print(f"    {cell:12s} {passed} of 3  -> {status[cell]}")

    for name, text, cell in (
        ("P2", "fpo-policy learns square under FPO++'s fine-tuning", "fpo-square"),
        ("P3", "fpo-policy learns square under DPPO's", "dppo-square"),
    ):
        s = status[cell]
        mark = (
            "NOT READ"
            if s == "not read"
            else ("HOLDS" if s == "learns" else "FALSIFIED")
        )
        print(f"\n{name}  {text}\n    {mark}")

    print("\nreported: success by window of ten iterations")
    lines = ["cell\tseed\titerations\tstart_success\tend_success\tgain"]
    for cell, (_, iters, start, _) in CELLS.items():
        for seed in SEEDS:
            s = rows[cell, seed]["success"]
            spans = [mean(s[a : a + 10]) for a in range(0, iters, 10)]
            print(
                f"    {cell:12s} seed {seed}  " + "  ".join(f"{v:.3f}" for v in spans)
            )
            a, b = mean(s[:start]), mean(s[iters - 10 : iters])
            lines.append(f"{cell}\t{seed}\t{len(s)}\t{a:.3f}\t{b:.3f}\t{b - a:+.3f}")
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
