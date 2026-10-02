"""E49: judge PROTOCOL.md's checks and predictions from a results directory.

    python summarise.py [DIR]      # default: results/

Every run is DIR/<trainer>-<env>-seed<s>/, as `run_bridged.sh` left it on
guangzhao: run.txt, trainer.log, episodes.csv (the bridge's record of every
episode), and the trainer's own record: trainer-logs/monitor.csv (SB3),
TensorBoard events (CleanRL under runs/, RLinf under trainer-logs/), and for
Pendulum the C++ clients' cpp-client-<i>.log.

A batch is the k-th episode of every slot. All 16 envs start together and
their episodes have a fixed length, so a batch ends on one step.
"""

from __future__ import annotations

import csv
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent

# env: (batches per run, episode length, the status rule's two windows, bar)
ENVS = {
    # E44's rule: iterations 1-2 against 16-25 of 4,096 frames, which are
    # episodes 1-2 against 20-32 of 200 steps each.
    "pendulum": dict(batches=32, length=200, first=(1, 2), last=(20, 32), bar=500),
    # E43's and E48's rule: the first batch against the mean of the last five.
    "halfcheetah": dict(batches=25, length=1000, first=(1, 1), last=(21, 25), bar=200),
}
CELLS = [
    ("rlinf", "pendulum"),
    ("sb3", "pendulum"),
    ("sb3", "halfcheetah"),
    ("cleanrl", "halfcheetah"),
]
# RLinf x Pendulum runs 409,600 steps (PROTOCOL.md, known item 4): 128 batches.
# E44's rule scaled with it: the first two batches against the last 40%.
# Its first 102,400 steps are also judged by the 32-batch rule, as reported.
OVERRIDES = {
    ("rlinf", "pendulum"): dict(
        batches=128, length=200, first=(1, 2), last=(77, 128), bar=500
    ),
}
SEEDS = (0, 1, 2)


def spec_for(trainer: str, env: str) -> dict:
    return OVERRIDES.get((trainer, env), ENVS[env])


def close(a: float, b: float) -> bool:
    # The trainers' own counts may sum float32 rewards; the bridge sums float64.
    return abs(a - b) <= 1e-4 * abs(b) + 0.01


def episodes(run: pathlib.Path) -> list[dict]:
    with open(run / "episodes.csv", encoding="utf-8", newline="") as f:
        return [
            {"step": int(r["step"]), "slot": int(r["slot"]), "episode": int(r["episode"]),
             "return": float(r["return"]), "length": int(r["length"])}
            for r in csv.DictReader(f)
        ]  # fmt: skip


def batch_means(rows: list[dict]) -> list[float]:
    by_k: dict[int, list[float]] = {}
    for r in rows:
        by_k.setdefault(r["episode"], []).append(r["return"])
    return [statistics.mean(by_k[k]) for k in sorted(by_k)]


def scalars(run: pathlib.Path, tag: str) -> list[float]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    values: list[float] = []
    for events in sorted(run.rglob("events.out.tfevents*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        if tag in acc.Tags()["scalars"]:
            values += [e.value for e in acc.Scalars(tag)]
    return values


def same_multiset(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(close(x, y) for x, y in zip(sorted(a), sorted(b)))


def trainer_agrees(
    trainer: str, run: pathlib.Path, rows: list[dict]
) -> tuple[bool, str]:
    """V2: the trainer's own record of the episodes is the bridge's."""
    bridge = [r["return"] for r in rows]
    if trainer == "sb3":
        with open(run / "trainer-logs" / "monitor.csv", encoding="utf-8") as f:
            next(f)  # VecMonitor's JSON header line
            own = [float(r["r"]) for r in csv.DictReader(f)]
        return same_multiset(own, bridge), f"monitor.csv {len(own)} episodes"
    if trainer == "cleanrl":
        own = scalars(run / "runs", "charts/episodic_return")
        return same_multiset(own, bridge), f"charts/episodic_return {len(own)} episodes"
    # RLinf logs one env/return per epoch (256 steps) in which episodes
    # ended: the mean over those episodes. Group the bridge's record the same
    # way. (For HalfCheetah an epoch holds at most one batch, so this is the
    # batch mean; a Pendulum epoch can hold two.)
    own = scalars(run / "trainer-logs", "env/return")
    by_epoch: dict[int, list[float]] = {}
    for r in rows:
        by_epoch.setdefault((r["step"] - 1) // 256, []).append(r["return"])
    means = [statistics.mean(by_epoch[k]) for k in sorted(by_epoch)]
    ok = len(own) == len(means) and all(close(a, b) for a, b in zip(own, means))
    return ok, f"env/return {len(own)} epochs, the bridge's {len(means)}"


def clients_agree(
    trainer: str, env: str, run: pathlib.Path, rows: list[dict]
) -> tuple[bool | None, str]:
    """V3, Pendulum only: each C++ client's own count of its episodes.

    A client prints its count when the server stops the run. RLinf never
    closes its environments (E48), so under RLinf the bridge sends no stop
    and the clients, killed by run_bridged.sh, print nothing: V3 does not
    apply there.
    """
    if env != "pendulum":
        return None, "no client-side count"
    if trainer == "rlinf":
        return None, "not applicable: RLinf ends without a stop (E48)"
    logs = sorted(run.glob("cpp-client-*.log"))
    counts, last10 = [], []
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"after (\d+) steps, (\d+) episodes", text)
        t = re.search(r"last \d+: (\S+)", text)
        if m and t:
            counts.append((int(m.group(1)), int(m.group(2))))
            last10.append(float(t.group(1)))
    per_slot: dict[int, list[float]] = {}
    for r in sorted(rows, key=lambda r: r["episode"]):
        per_slot.setdefault(r["slot"], []).append(r["return"])
    bridge_last10 = [statistics.mean(v[-10:]) for v in per_slot.values()]
    spec = ENVS[env]
    full = spec["batches"] * spec["length"]
    ok = (
        len(logs) == 16
        and counts == [(full, spec["batches"])] * 16
        # The client prints six significant digits.
        and len(last10) == len(bridge_last10)
        and all(
            abs(a - b) <= 0.01 for a, b in zip(sorted(last10), sorted(bridge_last10))
        )
    )
    return ok, f"{len(counts)} clients: {sorted(set(counts))}"


def judge(trainer: str, env: str, run: pathlib.Path) -> bool:
    spec = spec_for(trainer, env)
    rows = episodes(run)
    status = (run / "run.txt").read_text(encoding="utf-8")
    log = (run / "trainer.log").read_text(encoding="utf-8", errors="replace")
    bad = [w for w in ("Traceback", "ClientLost", "TimeoutError") if w in log]
    slots = sorted({r["slot"] for r in rows})
    per_slot = {s: sum(r["slot"] == s for r in rows) for s in slots}
    v1 = (
        "trainer rc=0" in status
        and not bad
        and slots == list(range(16))
        and set(per_slot.values()) == {spec["batches"]}
        and {r["length"] for r in rows} == {spec["length"]}
    )
    v2, v2_note = trainer_agrees(trainer, run, rows)
    v3, v3_note = clients_agree(trainer, env, run, rows)
    means = batch_means(rows)
    (a, b), (c, d) = spec["first"], spec["last"]
    first = statistics.mean(means[a - 1 : b])
    last = statistics.mean(means[c - 1 : d]) if len(means) >= d else float("nan")
    learns = last - first >= spec["bar"]
    if spec is not ENVS[env]:
        # Reported, not judged: the env's own rule over the first 102,400 steps.
        base = ENVS[env]
        (e, f), (g, h) = base["first"], base["last"]
        early = statistics.mean(means[g - 1 : h]) - statistics.mean(means[e - 1 : f])
        print(f"{run.name}: at 102,400 steps, batches {e}-{f} -> {g}-{h}: {early:+.0f}")
    print(
        f"{run.name}: V1 {v1} ({len(rows)} episodes{', ' + ', '.join(bad) if bad else ''}), "
        f"V2 {v2} ({v2_note}), V3 {v3} ({v3_note})\n"
        f"   batches {a}-{b} {first:.0f} -> {c}-{d} {last:.0f} ({last - first:+.0f}, bar +{spec['bar']}): "
        f"{'learns' if learns else 'does not learn'}\n"
        f"   batch means " + " ".join(f"{m:.0f}" for m in means)
    )  # fmt: skip
    return v1 and v2 and v3 is not False and learns


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    runs = {p.name: p for p in sorted(root.iterdir()) if (p / "episodes.csv").exists()}
    registered = True
    for trainer, env in CELLS:
        learned = 0
        found = 0
        for name, run in runs.items():
            if name.startswith(f"{trainer}-{env}-seed"):
                found += 1
                learned += judge(trainer, env, run)
        if found:
            print(
                f"{trainer} x {env}: learns, with every check passing, on {learned} of {found}\n"
            )
        registered &= found == len(SEEDS) and all(
            f"{trainer}-{env}-seed{s}" in runs for s in SEEDS
        )
        if found == len(SEEDS):
            print(
                f"P ({trainer} x {env} learns on >= 2 of 3): {'holds' if learned >= 2 else 'FAILS'}\n"
            )
    if not registered:
        print(
            "(not every registered run is here; predictions are judged per cell found)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
