"""Read E43's verdicts, in PROTOCOL.md's order.

    python summarise.py [results-dir]

Expects, under results/:
  local/ cross/                   the servers' runs and logs (server_side.sh)
  local.out cross.out             server_side.sh's own output
  local/client-seed{s}.log        the local arm's client logs
  local/client-summary-seed{s}.json
  cross-clients/                  client_side.sh's output, client logs, and
                                  client-summary-seed{s}.json from the laptop
  throughput.tsv ladder.tsv       Part B, from the laptop
Writes summary.tsv next to this file.
"""

import csv
import json
import pathlib
import re
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
SEEDS = (0, 1, 2)
ITERS = 100
GAIN = 200.0
TS_IP = "100.75.226.89"
PREDICTED_EXTRA_S = 99 * 4096 * 2.76e-3
IMAGE_BYTES = {
    48: 128 * 128 * 3,
    184: 224 * 224 * 3 + 112 * 112 * 3,
    588: 448 * 448 * 3,
}


def read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def rewards(arm: str, seed: int):
    run = R / arm / "fpo" / "fpo-policy" / f"halfcheetah-{arm}-seed{seed}"
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return run, []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    return run, acc.Scalars("rollout/reward") if "rollout/reward" in acc.Tags()[
        "scalars"
    ] else []


def client_rc(arm: str, seed: int):
    out = (
        read(R / "local.out")
        if arm == "local"
        else read(R / "cross-clients" / "client_side.out")
    )
    m = re.search(rf"seed {seed} client rc=(\d+)", out)
    return int(m.group(1)) if m else None


def client_log(arm: str, seed: int) -> str:
    return read(
        R / ("local" if arm == "local" else "cross-clients") / f"client-seed{seed}.log"
    )


def main() -> int:
    runs = {}
    print("P1  both arms run end to end")
    p1 = True
    for arm in ("local", "cross"):
        for seed in SEEDS:
            run, r = rewards(arm, seed)
            log = read(R / arm / f"server-seed{seed}.log")
            saves = (
                [p for p in run.iterdir() if p.name.isdigit()] if run.exists() else []
            )
            rc = client_rc(arm, seed)
            ok = (
                len(r) >= ITERS
                and len(saves) >= 5
                and bool(log)
                and "Traceback" not in log
                and rc == 0
            )
            p1 &= ok
            runs[(arm, seed)] = (r, log)
            print(
                f"    {arm:5s} seed {seed}  {'HOLDS' if ok else 'FAILS'}  ({len(r)} iterations, {len(saves)} checkpoints, client rc {rc})"
            )

    print("\nV1  the arms are what they say")
    configs = set()
    v1 = True
    for (arm, seed), (_, log) in runs.items():
        m = re.search(r"Algorithm: fpo, Config: (FPOAlgoConfig\(.*?\))\s*$", log, re.M)
        configs.add(m.group(1) if m else None)
        host = TS_IP if arm == "cross" else "127.0.0.1"
        connected = host in client_log(arm, seed)
        v1 &= connected
        print(
            f"    {arm:5s} seed {seed}  client log names {host}: {'yes' if connected else 'NO'}"
        )
    same = len(configs) == 1 and None not in configs
    v1 &= same
    print(f"    six server configurations identical: {'yes' if same else 'NO'}")
    print(f"    {'PASS' if v1 else 'FAIL'}")

    print(
        f"\nstatus  learns: iterations 91-100 at least +{GAIN:.0f} over iteration 1, on 2 of 3"
    )
    gains, spans = {}, {}
    lines = ["arm\tseed\tfirst\tit50\tlast10\tgain\tspan_s"]
    for arm in ("local", "cross"):
        for seed in SEEDS:
            r, _ = runs[(arm, seed)]
            if len(r) < ITERS:
                gains[(arm, seed)] = float("nan")
                spans[(arm, seed)] = float("nan")
                continue
            v = [e.value for e in r]
            first, last10 = v[0], statistics.mean(v[90:100])
            gains[(arm, seed)] = last10 - first
            spans[(arm, seed)] = r[99].wall_time - r[0].wall_time
            print(
                f"    {arm:5s} seed {seed}  {first:8.1f} -> {last10:8.1f}  ({last10 - first:+.1f})  it50 {v[49]:.1f}"
            )
            lines.append(
                f"{arm}\t{seed}\t{first:.1f}\t{v[49]:.1f}\t{last10:.1f}\t{last10 - first:+.1f}\t{spans[(arm, seed)]:.1f}"
            )
    ok_read = p1 and v1
    for name, arm in (
        ("P2  the cross arm learns", "cross"),
        ("P3  the local arm learns", "local"),
    ):
        n = sum(gains[(arm, s)] >= GAIN for s in SEEDS)
        verdict = "NOT READ" if not ok_read else ("HOLDS" if n >= 2 else "FALSIFIED")
        print(f"\n{name}\n    {n} of 3  {verdict}")

    print(
        f"\nP4  the cross arm's extra time, iteration 1 -> 100, against {PREDICTED_EXTRA_S:.0f} s (±25%)"
    )
    within = 0
    for seed in SEEDS:
        extra = spans[("cross", seed)] - spans[("local", seed)]
        hit = 0.75 * PREDICTED_EXTRA_S <= extra <= 1.25 * PREDICTED_EXTRA_S
        within += hit
        print(
            f"    seed {seed}  local {spans[('local', seed)]:.0f} s  cross {spans[('cross', seed)]:.0f} s  extra {extra:.0f} s  {'in' if hit else 'OUT'}"
        )
    print(
        f"    {within} of 3  {'NOT READ' if not ok_read else ('HOLDS' if within >= 2 else 'FALSIFIED')}"
    )

    print("\nV2  ladder runs complete")
    rows = list(
        csv.DictReader(open(R / "ladder.tsv", encoding="utf-8"), delimiter="\t")
    )
    cells = {}
    for row in rows:
        if int(row["exchanges"]) == 1000:
            cells.setdefault((row["rung"], int(row["payload_kib"])), []).append(row)
    v2 = all(
        len(cells.get((rung, k), [])) == 5
        for rung in ("lo", "ts", "phys")
        for k in (0, 48, 184, 588)
    )
    print(f"    every rung and payload has 5 valid runs: {'yes' if v2 else 'NO'}")

    def med(rung: str, k: int, field: str = "rtt_ms") -> float:
        return (
            statistics.median(float(x[field]) for x in cells.get((rung, k), []))
            if cells.get((rung, k))
            else float("nan")
        )

    print("\nP5  lo and ts agree within 0.25 ms")
    p5 = True
    for k in (0, 48, 184, 588):
        d = abs(med("ts", k) - med("lo", k))
        p5 &= d <= 0.25
        print(
            f"    {k:3d} KiB  lo {med('lo', k):.3f}  ts {med('ts', k):.3f}  |d| {d:.3f}"
        )
    print(f"    {'NOT READ' if not v2 else ('HOLDS' if p5 else 'FALSIFIED')}")

    tp = [
        float(x["mb_per_s"])
        for x in csv.DictReader(
            open(R / "throughput.tsv", encoding="utf-8"), delimiter="\t"
        )
    ]
    b = statistics.median(tp) * 1e6
    print(
        f"\nP6  phys = phys(states) + 2 x bytes / B, B = median throughput {b / 1e6:.1f} MB/s ({', '.join(f'{x:.1f}' for x in tp)})"
    )
    p6 = True
    base = med("phys", 0)
    for k, nbytes in IMAGE_BYTES.items():
        pred = base + 2 * nbytes / b * 1000
        got = med("phys", k)
        hit = 0.75 * pred <= got <= 1.25 * pred
        p6 &= hit
        print(
            f"    {k:3d} KiB  predicted {pred:6.2f} ms  measured {got:6.2f} ms  {'in' if hit else 'OUT'}"
        )
    print(f"    {'NOT READ' if not v2 else ('HOLDS' if p6 else 'FALSIFIED')}")

    print("\nreported: ladder medians, pack / rtt / unpack ms")
    for rung in ("lo", "ts", "phys"):
        for k in (0, 48, 184, 588):
            print(
                f"    {rung:4s} {k:3d} KiB  {med(rung, k, 'pack_ms'):.3f} / {med(rung, k):.3f} / {med(rung, k, 'unpack_ms'):.3f}"
            )
    print(
        "\nreported: env client timing per call (infer wait / env step / feedback send, ms)"
    )
    for arm, d in (("local", R / "local"), ("cross", R / "cross-clients")):
        for seed in SEEDS:
            p = d / f"client-summary-seed{seed}.json"
            if not p.exists():
                print(f"    {arm:5s} seed {seed}  no summary")
                continue
            t = json.loads(p.read_text())["timing"]
            print(
                f"    {arm:5s} seed {seed}  {1000 * t['infer_wait_s'] / t['infer_calls']:.3f} / "
                f"{1000 * t['env_step_s'] / t['env_steps']:.3f} / {1000 * t['feedback_s'] / t['feedback_calls']:.3f}"
            )
    print(
        f"\nreported: E10's ratio with phys at 184 KiB: {med('phys', 184):.1f} ms against pi0.5's 100 ms forward = {med('phys', 184) / 100:.0%}"
    )

    (HERE / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
