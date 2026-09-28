"""Read E30's verdicts from each arm's logs and tensorboards.

    python summarise.py [results-dir]

P1, then V1, then P2, P3 - PROTOCOL.md's reading order - then the
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
ARMS = ("base", "chunk4", "steps20", "net512", "all")
# Every element's entropy depends only on the noise schedule, not the weights:
# Level 1.0 gives -1.920 + ln 10 (E29); twenty flow steps give -0.004 (the pilot).
WIDE = -1.920 + math.log(10.0)
ENTROPY = {
    "base": WIDE,
    "chunk4": WIDE,
    "steps20": -0.004,
    "net512": WIDE,
    "all": -0.004,
}
ENTROPY_TOL = 0.005
CLOSES = 200.0  # gain, on 2 of 3 seeds
BAR = 500.0  # the coverage figure's Hopper bar: mean of iterations 91-100


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

    def closes(arm: str) -> bool | None:
        if not readable(arm):
            return None
        return sum(g >= CLOSES for g in gains(arm)) >= 2

    def verdict(name: str, text: str, holds: bool | None) -> None:
        if holds is None:
            mark = "NOT READ - an arm did not run, or its manipulation did not take"
        else:
            mark = "HOLDS" if holds else "FALSIFIED"
        print(f"\n{name}  {text}\n    {mark}")

    print(f"\ngains (closes most of the gap: at least +{CLOSES:.0f} on 2 of 3)")
    for arm in ARMS:
        c = closes(arm)
        label = "closes" if c else ("does not close" if c is False else "not read")
        print(f"    {arm:8s} {show(arm):28s} {label}")

    verdict("P2", "the network closes most of the gap: net512 does", closes("net512"))
    c4, s20 = closes("chunk4"), closes("steps20")
    verdict(
        "P3",
        "neither the chunk nor the step count does: chunk4 and steps20 do not",
        None if c4 is None or s20 is None else not c4 and not s20,
    )

    print(
        f"\nreported: the Hopper bar - iterations 91-100 at least {BAR:.0f} on 2 of 3"
    )
    for arm in ARMS:
        last = [mean(data[arm, s]["reward"][ITERS - 10 : ITERS]) for s in SEEDS]
        met = sum(v >= BAR for v in last)
        shown = ", ".join(f"{v:.1f}" for v in last)
        print(
            f"    {arm:8s} {shown:28s} {met} of 3  {'meets it' if met >= 2 else 'does not'}"
        )

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
