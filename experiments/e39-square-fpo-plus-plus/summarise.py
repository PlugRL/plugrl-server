"""Read E39's verdicts from its logs and tensorboards.

    python summarise.py [results-dir]

P1, V1, V2, the status rule, then P2 - PROTOCOL.md's reading order - and the
success, the critic's explained variance and the clipped fraction by window.
A seed's start is the mean `rollout/success` of iterations 1-2 (collected by
the clone unchanged, before and during the critic-only iteration); its end is
the mean of the last ten. E37's summarise.py with E39's one cell. Writes
`summary.tsv`.
"""

import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
SEEDS = (0, 1, 2)
GAIN = 0.2  # learns: end minus start, in success rate, on 2 of 3
CELL = "fpopp-square"
ITERS = 100
START = 2
# V2: the critic learns the return. Mean explained variance over iterations
# 11-20, on every seed, above this.
EV_BAR = 0.2
# What every server's config line must show.
WANTED = (
    "output_mode='u'",
    "cfm_loss_steps=4",
    "cfm_loss_dims=7",
    "cfm_loss_sum_over_steps=True",
    "cfm_loss_huber_delta=1.0",
    "ratio_per_sample=True",
    "n_critic_warmup_itrs=1",
    "clipping_epsilon=0.01",
    "learning_rate=1e-05",
    "critic_learning_rate=0.0001",
    "adam_eps=1e-05",
    "weight_decay=1e-06",
    "actor_adam_beta2=0.99",
    "max_grad_norm=25.0",
    "normalize_advantage_per_minibatch=True",
    "fpo_playground_trick=False",
    "reward_scaling=1.0",
    "treat_truncated_as_done=True",
    "value_loss_coeff=0.5",
    "restore='except-critic'",
    "freeze_obs_stats=True",
    "value_hidden_dims=(512, 256)",
)


def run_dir(seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / CELL).glob(f"fpo/fpo-policy/{CELL}-seed{seed}"))
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
    rows = {}
    print("P1  the cell runs end to end")
    out_path = RESULTS / f"{CELL}.out"
    out = (
        out_path.read_text(encoding="utf-8", errors="replace")
        if out_path.exists()
        else ""
    )
    exits = {
        int(s): int(rc) for s, rc in re.findall(r"seed (\d) finished rc=(\d+)", out)
    }
    runs_ok = True
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
        runs_ok &= ok
        rows[seed] = dict(
            log=log,
            success=success,
            ev=curve(run, "rollout/explained_variance"),
            clipped=curve(run, "fpo/clipped_ratio_mean"),
        )
        print(
            f"    seed {seed}  {'HOLDS' if ok else 'FAILS'}  ({len(success)} iterations)"
        )

    print("\nV1  every server started from the clone with FPO++'s square settings")
    missing = set()
    for seed in SEEDS:
        log = rows[seed]["log"]
        if "Restoring from" not in log or "with restore=except-critic" not in log:
            missing.add("the restore line")
        missing |= {w for w in WANTED if w not in log}
    v1 = not missing
    print(f"    {'PASS' if v1 else f'FAIL - missing {sorted(missing)}'}")

    print(
        f"\nV2  the critic learns: explained variance over iterations 11-20 above {EV_BAR} on every seed"
    )
    evs = [mean(rows[s]["ev"][10:20]) for s in SEEDS]
    v2 = all(e > EV_BAR for e in evs)
    print(
        "    "
        + "  ".join(f"seed {s} {e:.3f}" for s, e in zip(SEEDS, evs))
        + f"  {'PASS' if v2 else 'FAIL'}"
    )

    print(
        f"\nstatus  learns: success, mean of the last ten minus the start, at least +{GAIN:.1f} on 2 of 3"
    )
    gains = []
    for seed in SEEDS:
        s = rows[seed]["success"]
        a = mean(s[:START])
        b = mean(s[ITERS - 10 : ITERS]) if len(s) >= ITERS else float("nan")
        gains.append(b - a)
        print(f"    seed {seed}  {a:.3f} -> {b:.3f}  ({b - a:+.3f})")
    passed = sum(g >= GAIN for g in gains)
    status = (
        "not read"
        if not (runs_ok and v1)
        else ("learns" if passed >= 2 else "does not learn")
    )
    print(f"    {passed} of 3  -> {status}")

    mark = (
        "NOT READ"
        if status == "not read"
        else ("HOLDS" if status == "learns" else "FALSIFIED")
    )
    print(
        f"\nP2  fpo-policy learns square under FPO with FPO++'s square fine-tuning\n    {mark}"
    )

    print(
        "\nreported: by window of ten iterations - success / explained variance / clipped fraction"
    )
    lines = ["cell\tseed\titerations\tstart_success\tend_success\tgain"]
    for seed in SEEDS:
        r = rows[seed]
        for name in ("success", "ev", "clipped"):
            spans = [mean(r[name][a : a + 10]) for a in range(0, ITERS, 10)]
            print(f"    seed {seed} {name:8s} " + " ".join(f"{v:6.3f}" for v in spans))
        s = r["success"]
        a, b = mean(s[:START]), mean(s[ITERS - 10 : ITERS])
        lines.append(f"{CELL}\t{seed}\t{len(s)}\t{a:.3f}\t{b:.3f}\t{b - a:+.3f}")
    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
