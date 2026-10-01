"""E46: read results/ and judge PROTOCOL.md's checks and predictions.

    python summarise.py

Part A comes from results/ladder.tsv and results/throughput.tsv. Part B
comes from results/train.txt, which is compare_train.py's output from
guangzhao.
"""

from __future__ import annotations

import csv
import pathlib
import re
import statistics

R = pathlib.Path(__file__).resolve().parent / "results"
PAYLOADS = (0, 48, 184, 588)
LABEL = {0: "states only", 48: "48 KiB", 184: "184 KiB", 588: "588 KiB"}


def verdict(ok: bool) -> str:
    return "holds" if ok else "FAILS"


def main() -> int:
    rows = list(
        csv.DictReader((R / "ladder.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    tput = [
        float(r["mb_per_s"])
        for r in csv.DictReader(
            (R / "throughput.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    ]
    B = statistics.median(tput) * 1e6  # bytes per second

    def cell(rung: str, arm: str, kib: int) -> list[dict]:
        return [
            r for r in rows
            if r["rung"] == rung and r["arm"] == arm and int(r["payload_kib"]) == kib
        ]  # fmt: skip

    def med(rung: str, arm: str, kib: int, field: str = "rtt_ms") -> float:
        return statistics.median(float(r[field]) for r in cell(rung, arm, kib))

    print(
        f"throughput: {', '.join(f'{t:.1f}' for t in tput)} MB/s, median {B / 1e6:.1f}"
    )

    # V1, V2
    counts = {
        (rung, arm, kib): len(cell(rung, arm, kib))
        for rung in ("lo", "ts", "phys") for arm in ("v1", "v2") for kib in PAYLOADS
    }  # fmt: skip
    complete = all(n == 5 for n in counts.values()) and all(
        int(r["exchanges"]) == 1000 for r in rows
    )
    print(f"V1: {len(rows)} runs of 120 recorded, all 1,000 exchanges: {complete}")
    # The checker's verdict for each run is the last field of its line in
    # part_a.out; ladder.tsv keeps the client's numbers only.
    out = (R / "part_a.out").read_text(encoding="utf-8", errors="replace")
    run_lines = [
        line for line in out.splitlines() if re.match(r"(lo|ts|phys)\tv[12]\t", line)
    ]
    clean = sum("no violations" in line for line in run_lines)
    print(f"V1: the checker found no violations in {clean} of {len(run_lines)} runs")
    v2_small = all(int(r["infer_bytes"]) < 1024 for r in rows if r["arm"] == "v2")
    v1_full = all(
        int(r["infer_bytes"]) >= int(r["feedback_bytes"]) - 200
        for r in rows if r["arm"] == "v1"
    )  # fmt: skip
    print(
        f"V2: every v2 infer < 1 KiB: {v2_small}; every v1 infer the observation's size: {v1_full}"
    )

    print("\nmedian round trip, ms (v1 / v2)")
    print(f"{'payload':<12}" + "".join(f"{rung:>20}" for rung in ("lo", "ts", "phys")))
    for kib in PAYLOADS:
        print(
            f"{LABEL[kib]:<12}"
            + "".join(
                f"{med(rung, 'v1', kib):>11.3f} /{med(rung, 'v2', kib):>7.3f}"
                for rung in ("lo", "ts", "phys")
            )
        )

    s = statistics.median(
        [float(r["rtt_ms"]) for r in cell("phys", "v1", 0) + cell("phys", "v2", 0)]
    )
    print(
        f"\ns = {s:.3f} ms (phys, states only, both arms pooled); B = {B / 1e6:.1f} MB/s"
    )
    for kib in (48, 184, 588):
        v1, v2 = med("phys", "v1", kib), med("phys", "v2", kib)
        P = statistics.median(int(r["feedback_bytes"]) for r in cell("phys", "v1", kib))
        ratio = (v1 - s) / (v2 - s)
        saved, saved_pred = v1 - v2, P / B * 1e3
        v1_pred = s + 2 * P / B * 1e3
        tag = "" if kib != 48 else "   (reported only)"
        print(
            f"{LABEL[kib]:<8} ratio {ratio:.2f}  saved {saved:.2f} ms against P/B "
            f"{saved_pred:.2f} ({(saved / saved_pred - 1) * 100:+.0f}%)  v1 {v1:.2f} "
            f"against s+2P/B {v1_pred:.2f} ({(v1 / v1_pred - 1) * 100:+.0f}%){tag}"
        )
        if kib == 48:
            continue
        print(
            f"   P1 {verdict(1.6 <= ratio <= 2.4)}   P2 {verdict(abs(saved / saved_pred - 1) <= 0.25)}"
            f"   P3 {verdict(abs(v1 / v1_pred - 1) <= 0.25)}"
        )

    p4 = True
    for rung in ("lo", "ts"):
        for kib in PAYLOADS:
            d = med(rung, "v2", kib) - med(rung, "v1", kib)
            ok = d <= 0.25 if kib == 588 else abs(d) <= 0.25
            p4 &= ok
            print(f"P4 {rung:<2} {LABEL[kib]:<12} v2 - v1 = {d:+.3f} ms  {verdict(ok)}")
    print(f"P4 {verdict(p4)}")

    print("\npack / unpack ms and frame bytes, phys, v1 then v2")
    for kib in PAYLOADS:
        print(
            f"{LABEL[kib]:<12} pack {med('phys', 'v1', kib, 'pack_ms'):.3f} / "
            f"{med('phys', 'v2', kib, 'pack_ms'):.3f}  unpack "
            f"{med('phys', 'v1', kib, 'unpack_ms'):.3f} / {med('phys', 'v2', kib, 'unpack_ms'):.3f}"
            f"  infer {int(cell('phys', 'v1', kib)[0]['infer_bytes']):,} / "
            f"{int(cell('phys', 'v2', kib)[0]['infer_bytes']):,}  feedback "
            f"{int(cell('phys', 'v1', kib)[0]['feedback_bytes']):,}"
        )
    print(
        f"\nE10's ratio at 184 KiB against a 100 ms pi0.5 forward: v1 "
        f"{med('phys', 'v1', 184):.1f} ms = {med('phys', 'v1', 184):.0f}%, v2 "
        f"{med('phys', 'v2', 184):.1f} ms = {med('phys', 'v2', 184):.0f}%"
    )

    train = R / "train.txt"
    if train.exists():
        text = train.read_text(encoding="utf-8")
        print("\nPart B (results/train.txt)")
        iters = [int(m) for m in re.findall(r"(\d+) iterations", text)]
        resyncs = [int(m) for m in re.findall(r"resync closes (\d+)", text)]
        rc = re.findall(r"client rc=(\d+)", text)
        print(f"V3: iterations {iters}, resync closes {resyncs}, client rc {rc}")
        for line in text.splitlines():
            if "==" in line or "reused_observations" in line:
                print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
