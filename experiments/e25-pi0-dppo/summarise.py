"""Read E25's verdicts from the files copied back from the cluster.

    python summarise.py

P1, then P2, then the movements - PROTOCOL.md's reading order. Reads, from
`results/`: the cell's output (`dppo-libero.out`), the server's log, the
card-memory samples, the tensorboard, the evaluation harness's table, and
movement.py's output for DPPO's last checkpoint and for FPO's first
iteration (E14's checkpoint, the one E21 and E22 took their update from).
"""

import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ITERS = 2
CARD_MIB = 24576
ARM = "dppo-iter2"


def text(name: str) -> str:
    path = RESULTS / name
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def curve(tag: str) -> list[float]:
    events = sorted((RESULTS / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    return [e.value for e in acc.Scalars(tag)]


def movements(name: str) -> dict[str, float]:
    return {
        g: float(d)
        for g, d in re.findall(
            r"^\s+(\w+)\s+moved.*relative distance ([\d.]+)$", text(name), flags=re.M
        )
    }


def main() -> int:
    out = text("dppo-libero.out")
    server = text("server.log")

    print("P1  pi0.5 trains end to end under DPPO")
    learned = curve("losses/clipfrac")
    ck = re.search(r"checkpoints: (.*)$", out, flags=re.M)
    saves = sorted(
        int(s) for s in re.findall(r"/(\d+)/model\.safetensors", ck.group(1) if ck else "")
    )
    peak = re.search(r"peak MiB on cards 0 and 1: (\d+), (\d+)", out)
    card0 = int(peak.group(1)) if peak else None
    parts = {
        f"{ITERS} iterations logged ({len(learned)})": len(learned) >= ITERS,
        f"a checkpoint after each ({saves})": len(saves) >= ITERS,
        "no traceback in the server log": bool(server) and "Traceback" not in server,
        "the server exited on its own": "server exited after" in out
        and "clients exited while the server was running" not in out,
        f"card 0 under {CARD_MIB} MiB (peak {card0})": card0 is not None
        and card0 < CARD_MIB,
    }
    for part, ok in parts.items():
        print(f"    {'yes' if ok else 'NO ':3s}  {part}")
    p1 = all(parts.values())
    print(f"    {'HOLDS' if p1 else 'FALSIFIED'}")

    print("\nnull, an unperturbed actor: 30.9 +/- 3.6, range 28-37 (E15 correction 2)")
    print("FPO's first iteration: 0 of 50 (E14)")

    print("\nP2  DPPO's two iterations leave at least 20 of 50")
    rows = {}
    table = text("stageC_eval.tsv").splitlines()
    if table:
        header = table[0].split("\t")
        for line in table[1:]:
            row = dict(zip(header, line.split("\t")))
            if row.get("policy") != "policy":
                rows[row["policy"]] = row
    row = rows.get(ARM)
    if not p1:
        print("    not read - P1 does not hold")
    elif row is None or row["valid"] != "true" or row["episodes"] != "50":
        print(f"    NOT READ - {ARM} is missing or invalid")
    else:
        s = int(row["successes"])
        print(f"    {ARM}  {s} of 50  (95% CI {row['ci_lo']}-{row['ci_hi']})")
        if s >= 20:
            print("    HOLDS - DPPO's updates are too small to do FPO's damage in two iterations")
        elif s <= 5:
            print("    FALSIFIED - DPPO destroys it too; the damage is not FPO's alone")
        else:
            print("    6 to 19 - reported, no reading")

    print("\nreported: training, per iteration")
    for tag in ("rollout/reward", "losses/clipfrac", "losses/approx_kl"):
        values = curve(tag)
        print(f"    {tag:18s} {', '.join(f'{v:.4f}' for v in values) or 'not logged'}")

    print("\nreported: relative distance from the base, per module group")
    dppo = movements("movement.txt")
    fpo = movements("movement-fpo-e14.txt")
    print(f"    {'group':6s} {'DPPO, 2 iters':>14s} {'FPO, 1 iter':>12s} {'ratio':>7s}")
    for g in ("mod", "attn", "mlp", "io"):
        d, f = dppo.get(g), fpo.get(g)
        ratio = f"{d / f:.3f}" if d is not None and f else "-"
        print(
            f"    {g:6s} "
            f"{f'{100 * d:.3f}%' if d is not None else 'missing':>14s} "
            f"{f'{100 * f:.3f}%' if f is not None else 'missing':>12s} "
            f"{ratio:>7s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
