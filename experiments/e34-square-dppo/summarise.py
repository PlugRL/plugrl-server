"""Read E34's verdicts from the cell's logs and tensorboards.

    python summarise.py [results-dir]

P1, V1, the status rule, then P2 - PROTOCOL.md's reading order - and the
success, return and update diagnostics. The start is the mean success of
iterations 1-3, all collected by the released policy unchanged (the first
two iterations train only the critic); the end is the mean of the last ten.
Writes `summary.tsv`.
"""

import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
CELL = "dppo-square"
SEEDS = (0, 1, 2)
ITERS = 50
START = 3  # iterations 1-3: the released policy, unchanged
GAIN = 0.2  # learns: end minus start, in success rate, on 2 of 3


def run_dir(seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / CELL).glob(f"*/*/{CELL}-seed{seed}"))
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
    out_path = RESULTS / f"{CELL}.out"
    out = (
        out_path.read_text(encoding="utf-8", errors="replace")
        if out_path.exists()
        else ""
    )
    exits = {
        int(s): int(rc) for s, rc in re.findall(r"seed (\d) finished rc=(\d+)", out)
    }

    rows = {}
    print("P1  the cell runs end to end")
    for seed in SEEDS:
        run = run_dir(seed)
        log_path = RESULTS / CELL / f"server-seed{seed}.log"
        log = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        success = curve(run, "rollout/success")
        saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
        ok = (
            len(success) >= ITERS
            and len(saves) >= ITERS // 10
            and bool(log)
            and "Traceback" not in log
            and exits.get(seed) == 0
        )
        rows[seed] = dict(
            ok=ok,
            log=log,
            success=success,
            reward=curve(run, "rollout/reward"),
            length=curve(run, "rollout/length"),
            kl=curve(run, "losses/approx_kl"),
            clipfrac=curve(run, "losses/clipfrac"),
        )
        print(
            f"    seed {seed}  {'HOLDS' if ok else 'FAILS'}  ({len(success)} iterations)"
        )
    p1 = all(rows[s]["ok"] for s in SEEDS)

    print(
        "\nV1  every server started from the released checkpoint's ema, 10 of 20 steps"
    )
    v1 = True
    for seed in SEEDS:
        log = rows[seed]["log"]
        ema = "Loaded the ema weights of" in log
        ft = "ft_denoising_steps=10" in log
        v1 &= ema and ft
        print(
            f"    seed {seed}  ema {'yes' if ema else 'NO'}  ft 10 {'yes' if ft else 'NO'}"
        )
    print(f"    {'PASS' if v1 else 'FAIL - not read'}")

    print(
        f"\nstatus  learns: success, mean of the last ten minus iterations 1-{START}, "
        f"at least +{GAIN:.1f} on 2 of 3"
    )
    gains = {}
    for seed in SEEDS:
        s = rows[seed]["success"]
        start = mean(s[:START])
        end = mean(s[ITERS - 10 : ITERS]) if len(s) >= ITERS else float("nan")
        gains[seed] = end - start
        print(f"    seed {seed}  {start:.3f} -> {end:.3f}  ({end - start:+.3f})")
    passed = sum(g >= GAIN for g in gains.values())
    learns = p1 and v1 and passed >= 2
    print(
        f"    {passed} of 3  -> {'learns' if learns else 'does not learn in 50 iterations'}"
    )

    print("\nP2  dppo-policy learns square under DPPO's own fine-tuning")
    print(
        "    NOT READ"
        if not (p1 and v1)
        else f"    {'HOLDS' if learns else 'FALSIFIED'}"
    )

    print("\nreported: success by window of ten iterations")
    for seed in SEEDS:
        s = rows[seed]["success"]
        spans = [mean(s[a : a + 10]) for a in range(0, ITERS, 10)]
        print(f"    seed {seed}  " + "  ".join(f"{v:.3f}" for v in spans))

    print("\nreported: update diagnostics, mean over iterations 3-12 and the last ten")
    for seed in SEEDS:
        for tag in ("kl", "clipfrac"):
            c = rows[seed][tag]
            print(
                f"    seed {seed}  {tag:9s} {mean(c[2:12]):.2e} -> {mean(c[-10:]):.2e}"
            )

    lines = [
        "seed\titerations\truns\tstart_success\tend_success\tgain\tstart_return\tend_return"
    ]
    for seed in SEEDS:
        r = rows[seed]
        s, rew = r["success"], r["reward"]
        lines.append(
            "\t".join(
                [
                    str(seed),
                    str(len(s)),
                    str(r["ok"]).lower(),
                    f"{mean(s[:START]):.3f}",
                    f"{mean(s[ITERS - 10 : ITERS]):.3f}",
                    f"{gains[seed]:+.3f}",
                    f"{mean(rew[:START]):.1f}",
                    f"{mean(rew[ITERS - 10 : ITERS]):.1f}",
                ]
            )
        )
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
