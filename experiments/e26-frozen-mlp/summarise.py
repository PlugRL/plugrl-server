"""Read E26's verdicts from the files copied back from the cluster.

    python summarise.py

V1, V2, then P1, then P2, then `base` and the movements - PROTOCOL.md's
reading order. Reads `results/check-frozen.txt` (check_frozen.py's output,
control's checkpoint first and frozen's second, as run.sh calls it) and
`results/stageC_eval.tsv`, the evaluation harness's own table, whose `valid`
already requires the client to have exited 0.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ("base", "control", "frozen")
GROUPS = ("mod", "attn", "mlp", "io")
MLP_TENSORS = 54


def read_check() -> dict[str, dict[str, tuple[int, int, float]]]:
    """{arm: {group: (moved, unchanged, relative distance)}}, arm from the path."""
    path = RESULTS / "check-frozen.txt"
    arms: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.endswith("model.safetensors"):
            hit = re.search(r"/e26/(control|frozen)/", line)
            current = hit.group(1) if hit else None
            if current:
                arms[current] = {}
            continue
        m = re.match(
            r"^\s+(\w+)\s+moved\s+(\d+)\s+unchanged\s+(\d+)\s+relative distance ([\d.]+)$",
            line,
        )
        if m and current:
            arms[current][m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    return arms


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
    check = read_check()
    frozen, control = check.get("frozen", {}), check.get("control", {})
    v1_parts = {
        f"frozen: all {MLP_TENSORS} MLP tensors bit-identical to the base": frozen.get(
            "mlp", (None,)
        )[:2]
        == (0, MLP_TENSORS),
        "frozen: the other groups moved": all(
            frozen.get(g, (0,))[0] > 0 for g in ("mod", "attn", "io")
        ),
        "control: the MLP tensors moved": control.get("mlp", (0,))[0] > 0,
    }
    print("V1  the freeze held")
    for part, ok in v1_parts.items():
        print(f"    {'yes' if ok else 'NO ':3s}  {part}")
    v1 = all(v1_parts.values())
    print(f"    {'PASS' if v1 else 'FAIL - frozen is not what it claims and is not read'}")

    rows = read_rows()
    valid = {
        a: a in rows and rows[a]["valid"] == "true" and rows[a]["episodes"] == "50"
        for a in ARMS
    }
    print("\nV2  every evaluation valid")
    for a in ARMS:
        print(f"    {a:8s} {'valid' if valid[a] else 'MISSING OR INVALID'}")

    def score(arm: str) -> int | None:
        return int(rows[arm]["successes"]) if valid[arm] else None

    print("\nnull, an unperturbed actor on the older code: 30.9 +/- 3.6, range 28-37")
    print("FPO's first iteration on the older code: 0 of 50 (E14); E22 only-mlp: 0")

    print("\nP1  the collapse reproduces on this code: control at most 5 of 50")
    c = score("control")
    p1 = False
    if c is None:
        print("    NOT READ - control's evaluation is missing or invalid")
    else:
        print(f"    control  {c} of 50")
        p1 = c <= 5
        print(
            "    HOLDS"
            if p1
            else "    FALSIFIED - the collapse is not a property of FPO on this "
            "policy as the code stands; frozen is not read as a remedy"
        )

    print("\nP2  freezing the MLP prevents it: frozen at least 20 of 50")
    f = score("frozen")
    if not p1:
        print("    not read - P1 does not hold")
    elif not v1:
        print("    not read - V1 failed")
    elif f is None:
        print("    NOT READ - frozen's evaluation is missing or invalid")
    else:
        print(f"    frozen   {f} of 50")
        if f >= 20:
            print("    HOLDS - a located cause with a remedy")
        elif f <= 5:
            print("    FALSIFIED - the rest of the expert, trained, destroys it too")
        else:
            print("    6 to 19 - partial, reported, no reading")

    print("\nreported: base on this code, against the older null")
    b = score("base")
    print(f"    base     {b if b is not None else 'not read'} of 50")

    print("\nreported: relative distance from the base, per module group")
    print(f"    {'group':6s} {'control':>18s} {'frozen':>18s}")
    for g in GROUPS:
        cells = []
        for arm in (control, frozen):
            if g in arm:
                moved, still, d = arm[g]
                cells.append(f"{100 * d:.3f}% ({moved}/{moved + still})")
            else:
                cells.append("missing")
        print(f"    {g:6s} {cells[0]:>18s} {cells[1]:>18s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
