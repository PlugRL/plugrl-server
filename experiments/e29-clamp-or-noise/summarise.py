"""Read E29's verdicts from each arm's logs and tensorboards.

    python summarise.py [results-dir]

P1, then V1, then P2, P3, P4 - PROTOCOL.md's reading order - then the
update diagnostics. The gain is the mean return of iterations 91-100 minus
the first iteration's. Writes `summary.tsv`, one row per arm and seed.
"""

import datetime
import math
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
SEEDS = (0, 1, 2)
ITERS = 100
ARMS = ("control", "noclamp", "wide")
# Every element's entropy depends only on the noise schedule, not the weights:
# E28 logged -1.920 at level 0.1, and scaling the level by k adds ln k.
ENTROPY = {"control": -1.920, "noclamp": -1.920, "wide": -1.920 + math.log(10.0)}
ENTROPY_TOL = 0.005
CONTROL_MAX = 50.0
LEARNS = 100.0


def run_dir(arm: str, seed: int) -> pathlib.Path | None:
    hits = sorted((RESULTS / arm).glob(f"*/*/{arm}-seed{seed}"))
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


def actor_movement(run: pathlib.Path | None) -> float:
    """Relative distance of the actor's weights, first checkpoint to last
    (iterations 20 and 100): how far eighty iterations of DPPO moved it."""
    from safetensors import safe_open

    saves = (
        sorted(
            (p for p in run.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
        )
        if run
        else []
    )
    if len(saves) < 2:
        return float("nan")
    num = den = 0.0
    with (
        safe_open(str(saves[0] / "model.safetensors"), "pt") as a,
        safe_open(str(saves[-1] / "model.safetensors"), "pt") as b,
    ):
        for key in a.keys():
            if key.startswith("actor."):
                x, y = a.get_tensor(key).double(), b.get_tensor(key).double()
                num += float((y - x).pow(2).sum())
                den += float(x.pow(2).sum())
    return math.sqrt(num / den) if den else float("nan")


def main() -> int:
    data = {}
    print("P1  every arm runs end to end")
    runs_ok = {}
    for arm in ARMS:
        out_path = RESULTS / f"{arm}.out"
        out = (
            out_path.read_text(encoding="utf-8", errors="replace")
            if out_path.exists()
            else ""
        )
        exits = {
            int(s): int(rc) for s, rc in re.findall(r"seed (\d) finished rc=(\d+)", out)
        }
        ok_arm = True
        for seed in SEEDS:
            run = run_dir(arm, seed)
            reward = curve(run, "rollout/reward")
            saves = [p for p in run.iterdir() if p.name.isdigit()] if run else []
            log_path = RESULTS / arm / f"server-seed{seed}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            ok = (
                len(reward) >= ITERS
                and len(saves) >= ITERS // 20
                and bool(log)
                and "Traceback" not in log
                and exits.get(seed) == 0
            )
            ok_arm &= ok
            gain = (
                mean(reward[ITERS - 10 : ITERS]) - reward[0]
                if len(reward) >= ITERS
                else float("nan")
            )
            data[arm, seed] = dict(
                ok=ok,
                reward=reward,
                gain=gain,
                entropy=curve(run, "losses/entropy"),
                kl=curve(run, "losses/approx_kl"),
                clipfrac=curve(run, "losses/clipfrac"),
                length=curve(run, "rollout/length"),
            )
        runs_ok[arm] = ok_arm
        minutes = wall_minutes(out)
        print(
            f"    {arm:8s} {'HOLDS' if ok_arm else 'FAILS'}"
            + (f"   ({minutes:.0f} min)" if minutes is not None else "")
        )

    print("\nV1  the manipulation: each arm's logged entropy, every iteration")
    v1 = {}
    for arm in ARMS:
        values = [v for s in SEEDS for v in data[arm, s]["entropy"]]
        v1[arm] = bool(values) and all(
            abs(v - ENTROPY[arm]) <= ENTROPY_TOL for v in values
        )
        span = f"{min(values):.4f} to {max(values):.4f}" if values else "not logged"
        print(
            f"    {arm:8s} expected {ENTROPY[arm]:+.4f}, logged {span}  "
            f"{'PASS' if v1[arm] else 'FAIL'}"
        )

    def gains(arm: str) -> list[float]:
        return [data[arm, s]["gain"] for s in SEEDS]

    def show(arm: str) -> str:
        return ", ".join(f"{g:+.1f}" for g in gains(arm))

    def readable(arm: str) -> bool:
        return runs_ok[arm] and v1[arm]

    print(
        f"\nP2  the control reproduces E28: gain at most +{CONTROL_MAX:.0f} on every seed"
    )
    print(f"    control  {show('control')}")
    p2 = readable("control") and all(g <= CONTROL_MAX for g in gains("control"))
    print(
        f"    {'HOLDS' if p2 else 'FALSIFIED or not readable - nothing below is read'}"
    )

    for name, arm, text in (
        ("P3", "noclamp", "lifting the clamp alone lets it learn"),
        ("P4", "wide", "ten times the noise lets it learn"),
    ):
        print(f"\n{name}  {text}: gain at least +{LEARNS:.0f} on 2 of 3")
        print(f"    {arm:8s} {show(arm)}")
        if not p2:
            print("    not read - P2 does not hold")
        elif not readable(arm):
            print(
                "    NOT READ - the arm did not run, or its manipulation did not take"
            )
        else:
            passed = sum(g >= LEARNS for g in gains(arm))
            print(f"    {passed} of 3  {'HOLDS' if passed >= 2 else 'FALSIFIED'}")

    print("\nreported: update diagnostics, mean over iterations 1-10 and 91-100")
    for arm in ARMS:
        for tag in ("kl", "clipfrac"):
            first = mean([v for s in SEEDS for v in data[arm, s][tag][:10]])
            last = mean(
                [v for s in SEEDS for v in data[arm, s][tag][ITERS - 10 : ITERS]]
            )
            print(f"    {arm:8s} {tag:9s} {first:.2e} -> {last:.2e}")

    print("\nreported: the actor's relative movement, iteration 20 to 100, per seed")
    for arm in ARMS:
        moves = [actor_movement(run_dir(arm, s)) for s in SEEDS]
        print(f"    {arm:8s} {', '.join(f'{100 * m:.2f}%' for m in moves)}")

    print(
        "\nreported: first iteration and mean of the last ten, return and episode length"
    )
    lines = [
        "arm\tseed\titerations\truns\tfirst_return\tlast10_return\tgain\tfirst_length\tlast10_length"
    ]
    for arm in ARMS:
        for s in SEEDS:
            d = data[arm, s]
            reward, length = d["reward"], d["length"]
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
                f"    {arm:8s} seed {s}  return {first:7.1f} -> {last10:7.1f}   "
                f"length {first_len:6.1f} -> {last10_len:6.1f}"
            )
            lines.append(
                "\t".join(
                    [
                        arm,
                        str(s),
                        str(len(reward)),
                        str(d["ok"]).lower(),
                        f"{first:.1f}",
                        f"{last10:.1f}",
                        f"{d['gain']:.1f}",
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
