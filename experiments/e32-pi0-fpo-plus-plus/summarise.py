"""Read E32's verdicts from the files copied back from the cluster.

    python summarise.py

V1, V2, V3, then P1 to P4, then `base`, the movements and the learn-step
statistics - PROTOCOL.md's reading order. Reads, under `results/`:
`movement-iteration1.txt` and `movement.txt` (movement.py's output for the
first and the last checkpoints), `configs.txt` (each arm's config line),
`learn-stats.txt` (tb_read.py's output per arm) and `stageC_eval.tsv`, the
evaluation harness's own table, whose `valid` already requires the client to
have exited 0.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ("control", "chunk", "persample", "fpopp")
GROUPS = ("mod", "attn", "mlp", "io", "expert")
HOLDS, COLLAPSES = 20, 5
E15_SMALLEST_DESTRUCTIVE = (
    0.0066  # E15's lr-1e-6 arm: the expert moved this far and scored 0
)

# What each arm's config line must show, and must not.
CHUNK = (
    "output_mode='u'",
    "discretize_t_for_training=False",
    "cfm_loss_steps=5",
    "cfm_loss_dims=7",
    "cfm_loss_sum_over_steps=True",
)
VANILLA = (
    "output_mode='u_but_supervise_as_eps'",
    "discretize_t_for_training=True",
    "cfm_loss_steps=None",
    "cfm_loss_dims=None",
    "cfm_loss_sum_over_steps=False",
)
EXPECTED = {
    "control": VANILLA + ("ratio_per_sample=False",),
    "chunk": CHUNK + ("ratio_per_sample=False",),
    "persample": VANILLA + ("ratio_per_sample=True",),
    "fpopp": CHUNK + ("ratio_per_sample=True",),
}
COMMON = ("n_critic_warmup_itrs=1", "learning_rate=1e-05", "clipping_epsilon=0.05")


def read_movement(name: str) -> dict[str, dict[str, tuple[int, int, float]]]:
    """{arm: {group: (moved, unchanged, relative distance)}}, arm from the path."""
    path = RESULTS / name
    arms: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.endswith("model.safetensors"):
            hit = re.search(r"/e32/(\w+)/ck/", line)
            current = hit.group(1) if hit else None
            if current:
                arms[current] = {}
            continue
        m = re.match(
            r"^\s+(\w+)\s+moved\s+(\d+)\s+unchanged\s+(\d+)\s+relative distance ([\d.]+)$",
            line,
        )
        if m and current:
            arms[current][m.group(1)] = (
                int(m.group(2)),
                int(m.group(3)),
                float(m.group(4)),
            )
    return arms


def read_configs() -> dict[str, str]:
    path = RESULTS / "configs.txt"
    configs, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.startswith("## "):
            current = line[3:].strip()
        elif current and "FPOAlgoConfig(" in line:
            configs[current] = line
    return configs


def read_stats() -> dict[str, dict[str, float]]:
    """{arm: {tag: last value}} - the last learn step is the policy update."""
    path = RESULTS / "learn-stats.txt"
    stats, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.startswith("## "):
            current = line[3:].strip()
            stats[current] = {}
            continue
        m = re.match(r"^\s+fpo/(\S+)\s+n=(\d+)\s+first=\S+\s+last=(\S+)", line)
        if m and current:
            stats[current][m.group(1)] = float(m.group(3))
            stats[current]["_n"] = int(m.group(2))
    return stats


def read_rows() -> dict[str, dict]:
    path = RESULTS / "stageC_eval.tsv"
    rows = {}
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        if row.get("policy") != "policy":  # a header written twice by two evaluations
            rows[row["policy"]] = row
    return rows


def main() -> int:
    first = read_movement("movement-iteration1.txt")
    print("V1  the first iteration left each arm's actor as it was")
    v1 = {}
    for arm in ARMS:
        groups = first.get(arm, {})
        v1[arm] = bool(groups) and all(groups.get(g, (1,))[0] == 0 for g in GROUPS)
        print(f"    {arm:10s} {'PASS' if v1[arm] else 'FAIL - not read'}")

    configs = read_configs()
    print("\nV2  every arm's flags took")
    v2 = {}
    for arm in ARMS:
        line = configs.get(arm, "")
        missing = [s for s in EXPECTED[arm] + COMMON if s not in line]
        v2[arm] = bool(line) and not missing
        print(
            f"    {arm:10s} "
            + ("PASS" if v2[arm] else f"FAIL - missing {missing or 'the config line'}")
        )

    rows = read_rows()
    valid = {
        a: a in rows and rows[a]["valid"] == "true" and rows[a]["episodes"] == "50"
        for a in ("base",) + ARMS
    }
    print("\nV3  every evaluation valid")
    for a in ("base",) + ARMS:
        print(f"    {a:10s} {'valid' if valid[a] else 'MISSING OR INVALID'}")

    def score(arm: str) -> int | None:
        return int(rows[arm]["successes"]) if valid[arm] else None

    def readable(arm: str) -> bool:
        return v1[arm] and v2[arm] and valid[arm]

    print("\nnull, an unperturbed actor on the older code: 30.9 +/- 3.6, range 28-37")
    print("E26 on today's code: base 29, one vanilla FPO iteration 0")

    def verdict(name: str, text: str, arm: str, want: str) -> bool | None:
        print(f"\n{name}  {text}")
        if not readable(arm):
            print(f"    NOT READ - {arm} failed V1, V2 or V3")
            return None
        s = score(arm)
        held = s >= HOLDS
        collapsed = s <= COLLAPSES
        print(f"    {arm:10s} {s} of 50")
        ok = held if want == "holds" else collapsed
        if ok:
            print("    HOLDS")
        elif held or collapsed:
            print("    FALSIFIED")
        else:
            print(f"    {COLLAPSES + 1} to {HOLDS - 1} - partial, reported, no reading")
        return ok

    p1 = verdict(
        "P1", "the collapse reproduces: control at most 5", "control", "collapses"
    )
    if p1 is False:
        print("    the collapse is not a property of this code; P2 to P4 not read")
    else:
        verdict("P2", "FPO++ prevents it: fpopp at least 20", "fpopp", "holds")
        verdict("P3", "its chunk loss alone does: chunk at least 20", "chunk", "holds")
        verdict(
            "P4",
            "its per-sample ratio alone does not: persample at most 5",
            "persample",
            "collapses",
        )

    print("\nreported: base on this code")
    b = score("base")
    print(f"    base       {b if b is not None else 'not read'} of 50")

    last = read_movement("movement.txt")
    print("\nreported: relative distance from the base after iteration 2, per group")
    print("    " + f"{'group':7s}" + "".join(f"{a:>13s}" for a in ARMS))
    for g in GROUPS:
        cells = []
        for arm in ARMS:
            d = last.get(arm, {}).get(g)
            cells.append(f"{100 * d[2]:.3f}%" if d else "missing")
        print("    " + f"{g:7s}" + "".join(f"{c:>13s}" for c in cells))
    for arm in ARMS:
        d = last.get(arm, {}).get("expert")
        s = score(arm)
        if d and s is not None and s >= HOLDS and d[2] < E15_SMALLEST_DESTRUCTIVE:
            print(
                f"    {arm} held having moved the expert {100 * d[2]:.3f}%, less than "
                f"any destructive run so far ({100 * E15_SMALLEST_DESTRUCTIVE:.2f}%)"
            )

    stats = read_stats()
    print("\nreported: the policy update's learn-step statistics (iteration 2)")
    tags = (
        "initial_cfm_loss_mean",
        "cfm_loss_mean",
        "policy_ratio_mean",
        "clipped_ratio_mean",
    )
    print("    " + f"{'arm':10s}" + "".join(f"{t:>24s}" for t in tags))
    for arm in ARMS:
        s = stats.get(arm, {})
        print(
            "    "
            + f"{arm:10s}"
            + "".join(f"{s[t]:>24.6g}" if t in s else f"{'missing':>24s}" for t in tags)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
