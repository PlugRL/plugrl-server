"""Read E44's verdicts, in PROTOCOL.md's order.

    python summarise.py [results-dir]

Expects results/cpp and results/python from `run.sh`, with cpp.out and
python.out beside them. Writes summary.tsv next to this file.
"""

import pathlib
import re
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
SEEDS = (1, 2, 3)
ITERS = 25
GAIN = 500.0
SPREAD = 150.0
SETTINGS = (
    "learning_rate=0.001",
    "anneal_lr=False",
    "buffer_size=4096",
    "batch_size=64",
    "update_epochs=10",
    "gamma=0.9",
    "gae_lambda=0.95",
    "clip_coef=0.2",
    "clip_vloss=False",
    "ent_coef=0.0",
    "normalize_rewards=False",
)


def read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def returns(arm: str, seed: int):
    run = R / arm / "ppo" / "gaussian-policy" / f"pendulum-{arm}-seed{seed}"
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    return acc.Scalars("rollout/reward") if "rollout/reward" in tags else []


def main() -> int:
    data = {}
    print("P1  both arms run end to end")
    p1 = True
    for arm in ("cpp", "python"):
        out = read(R / f"{arm}.out")
        for seed in SEEDS:
            r = returns(arm, seed)
            slog = read(R / arm / f"server-seed{seed}.log")
            clog = read(R / arm / f"client-seed{seed}.log")
            m = re.search(rf"seed {seed} client rc=(\d+)", out)
            rc = int(m.group(1)) if m else None
            ok = len(r) >= ITERS and bool(slog) and "Traceback" not in slog and rc == 0
            p1 &= ok
            data[(arm, seed)] = (r, slog, clog)
            print(
                f"    {arm:6s} seed {seed}  {'HOLDS' if ok else 'FAILS'}  ({len(r)} iterations, client rc {rc})"
            )

    print("\nV1  the arms are what they say")
    v1 = True
    configs = set()
    for (arm, seed), (_, slog, clog) in data.items():
        if arm == "cpp":
            right = (
                "connected to ws://127.0.0.1" in clog
                and "server stopped the run" in clog
            )
        else:
            right = "plugrl_env_client" in clog and "connected to ws://" not in clog
        m = re.search(r"Algorithm: ppo, Config: (PPOAlgoConfig\(.*?\))\s*$", slog, re.M)
        cfg = m.group(1) if m else ""
        configs.add(cfg)
        has = all(s in cfg for s in SETTINGS)
        v1 &= right and has
        print(
            f"    {arm:6s} seed {seed}  client {'yes' if right else 'NO'}  settings {'yes' if has else 'NO'}"
        )
    same = len(configs) == 1
    v1 &= same
    print(f"    six configurations identical: {'yes' if same else 'NO'}")
    print(f"    {'PASS' if v1 else 'FAIL'}")

    print(
        f"\nstatus  learns: iterations 16-25 at least +{GAIN:.0f} over iterations 1-2, on 2 of 3"
    )
    last = {"cpp": [], "python": []}
    learned = {}
    lines = ["arm\tseed\tfirst2\tit10\tlast10\tgain\twall_s"]
    for arm in ("cpp", "python"):
        n = 0
        for seed in SEEDS:
            r, _, _ = data[(arm, seed)]
            if len(r) < ITERS:
                continue
            v = [e.value for e in r]
            first, end = statistics.mean(v[:2]), statistics.mean(v[15:25])
            wall = r[ITERS - 1].wall_time - r[0].wall_time
            last[arm].append(end)
            n += end - first >= GAIN
            print(
                f"    {arm:6s} seed {seed}  {first:7.0f} -> {end:6.0f}  ({end - first:+.0f})  it10 {v[9]:.0f}"
            )
            lines.append(
                f"{arm}\t{seed}\t{first:.1f}\t{v[9]:.1f}\t{end:.1f}\t{end - first:+.1f}\t{wall:.1f}"
            )
        learned[arm] = n
    read_ok = p1 and v1
    for name, arm in (
        ("P2  the C++ arm learns", "cpp"),
        ("P3  the control learns", "python"),
    ):
        verdict = (
            "NOT READ"
            if not read_ok
            else ("HOLDS" if learned[arm] >= 2 else "FALSIFIED")
        )
        print(f"\n{name}\n    {learned[arm]} of 3  {verdict}")

    print(f"\nP4  every C++ seed within the control's range, widened by {SPREAD:.0f}")
    if last["python"] and last["cpp"]:
        lo, hi = min(last["python"]) - SPREAD, max(last["python"]) + SPREAD
        inside = [lo <= x <= hi for x in last["cpp"]]
        print(
            f"    control range {min(last['python']):.0f} to {max(last['python']):.0f}, widened {lo:.0f} to {hi:.0f}"
        )
        print("    cpp " + ", ".join(f"{x:.0f}" for x in last["cpp"]))
        verdict = (
            "NOT READ"
            if not read_ok or len(inside) < 3
            else ("HOLDS" if all(inside) else "FALSIFIED")
        )
        print(f"    {verdict}")

    print("\nreported: the C++ client's own count against the server's")
    for seed in SEEDS:
        _, _, clog = data[("cpp", seed)]
        m = re.search(r"after (\d+) steps, (\d+) episodes", clog)
        k = re.search(r"last \d+: (-?[\d.]+)", clog)
        print(
            f"    seed {seed}  {m.group(1) if m else '?'} steps, {m.group(2) if m else '?'} episodes; last ten episodes {k.group(1) if k else '?'}"
        )

    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
