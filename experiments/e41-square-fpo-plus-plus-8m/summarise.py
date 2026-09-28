"""Read E41's verdicts from its logs and tensorboards, and E39's.

    python summarise.py [results-dir] [e39-run-dir]

P1, V1, the status rule, then P2 - PROTOCOL.md's reading order - and the
success by window across both runs. A seed's start is E39's: the mean
`rollout/success` of E39's iterations 1-2. Its end is the mean over the last
ten of the 167 iterations, which are E41's. Writes `summary.tsv`.
"""

import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
E39 = (
    pathlib.Path(sys.argv[2]).resolve()
    if len(sys.argv) > 2
    else HERE.parents[2]
    / "e39-reg/experiments/e39-square-fpo-plus-plus/results/fpopp-square/fpo/fpo-policy"
)
SEEDS = (0, 1, 2)
GAIN = 0.2
CELL = "fpopp-square-8m"
TOTAL = 167  # iterations, E39's 100 and E41's 67
RESUMED = 100
START = 2


def tb(run: pathlib.Path | None, tag: str) -> list[float]:
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
    print("P1  the resumed cell runs end to end")
    p1 = True
    for seed in SEEDS:
        hits = sorted((RESULTS / CELL).glob(f"fpo/fpo-policy/{CELL}-seed{seed}"))
        run = hits[0] if hits else None
        log_path = RESULTS / CELL / f"server-seed{seed}.log"
        log = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        e39 = E39 / f"fpopp-square-seed{seed}"
        rows[seed] = dict(
            log=log,
            before=tb(e39, "rollout/success"),
            after=tb(run, "rollout/success"),
            itrs=tb(run, "train/train_itrs"),
            ev=tb(e39, "rollout/explained_variance")
            + tb(run, "rollout/explained_variance"),
        )
        saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
        ok = (
            len(rows[seed]["after"]) >= TOTAL - RESUMED
            and len(saves) >= (TOTAL - RESUMED) // 10
            and bool(log)
            and "Traceback" not in log
            and exits.get(seed) == 0
        )
        p1 &= ok
        print(
            f"    seed {seed}  {'HOLDS' if ok else 'FAILS'}  ({len(rows[seed]['after'])} iterations resumed)"
        )

    print("\nV1  every seed resumed its own E39 run in full and ran E39's settings")
    v1 = True
    for seed in SEEDS:
        r = rows[seed]
        resumed = (
            "with restore=all" in r["log"] and f"fpopp-square-seed{seed}" in r["log"]
        )
        counted = bool(r["itrs"]) and r["itrs"][0] == RESUMED + 1
        settings = all(
            w in r["log"]
            for w in (
                "critic_learning_rate=0.0001",
                "max_grad_norm=25.0",
                "normalize_advantage_per_minibatch=True",
                "cfm_loss_huber_delta=1.0",
                "fpo_playground_trick=False",
                "reward_scaling=1.0",
                "freeze_obs_stats=True",
            )
        )
        v1 &= resumed and counted and settings
        print(
            f"    seed {seed}  restore all {'yes' if resumed else 'NO'}  "
            f"first iteration {r['itrs'][0] if r['itrs'] else 'none'}  settings {'yes' if settings else 'NO'}"
        )
    print(f"    {'PASS' if v1 else 'FAIL - not read'}")

    print(
        f"\nstatus  learns: success over iterations {TOTAL - 9}-{TOTAL} minus E39's 1-{START}, at least +{GAIN:.1f} on 2 of 3"
    )
    gains = {}
    for seed in SEEDS:
        r = rows[seed]
        start = mean(r["before"][:START])
        end = (
            mean(r["after"][-10:])
            if len(r["after"]) >= TOTAL - RESUMED
            else float("nan")
        )
        gains[seed] = end - start
        print(f"    seed {seed}  {start:.3f} -> {end:.3f}  ({end - start:+.3f})")
    passed = sum(g >= GAIN for g in gains.values())
    status = (
        "not read"
        if not (p1 and v1)
        else ("learns" if passed >= 2 else "does not learn in 8M steps")
    )
    print(f"    {passed} of 3  -> {status}")
    mark = (
        "NOT READ"
        if status == "not read"
        else ("HOLDS" if status == "learns" else "FALSIFIED")
    )
    print(
        f"\nP2  fpo-policy learns square under FPO++'s fine-tuning in FPO++'s 8M steps\n    {mark}"
    )

    print("\nreported: success by window of ten iterations, E39 then E41")
    lines = ["seed\tstart\tend\tgain"]
    for seed in SEEDS:
        r = rows[seed]
        s = r["before"][:RESUMED] + r["after"]
        spans = [mean(s[a : a + 10]) for a in range(0, len(s), 10)]
        print(f"    seed {seed}  " + " ".join(f"{v:.3f}" for v in spans))
        lines.append(
            f"{seed}\t{mean(r['before'][:START]):.3f}\t{mean(r['after'][-10:]):.3f}\t{gains[seed]:+.3f}"
        )
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
