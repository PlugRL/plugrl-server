"""Read E22's verdicts from the evaluation harness's own table.

    python summarise.py

Reads `results/stageC_eval.tsv`, copied from the cluster, and prints V5, then
P1, then `keep-in-s1` if P1 holds, then P2, then the other localisation arms -
PROTOCOL.md's reading order. V1 to V4 are make_arms.py's exit status, recorded
in `results/construction.txt`.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TABLE = HERE / "results" / "stageC_eval.tsv"
CONSTRUCTION = HERE / "results" / "construction.txt"

ROT = ["rot-s1", "rot-s2", "rot-s3"]
ONLY = ["only-mod", "only-attn", "only-mlp", "only-io"]
ALL = ROT + ["keep-in-s1"] + ONLY


def read_rows() -> dict[str, dict]:
    rows = {}
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        cells = line.split("\t")
        if cells[0] == "policy":  # a header written twice by two evaluations at once
            continue
        row = dict(zip(header, cells))
        rows[row["policy"]] = row
    return rows


def main() -> int:
    construction = (
        CONSTRUCTION.read_text(encoding="utf-8") if CONSTRUCTION.exists() else ""
    )
    if "V1 V2 V3 V4 pass" not in construction:
        print(
            "V1-V4  construction checks did not pass, or were not recorded - nothing is read"
        )
        return 1
    print("V1-V4  pass (results/construction.txt)")

    rows = read_rows()
    missing = [a for a in ALL if a not in rows]
    invalid = [
        a
        for a in ALL
        if a in rows and (rows[a]["valid"] != "true" or rows[a]["episodes"] != "50")
    ]
    print(
        f"V5     {len(ALL) - len(missing) - len(invalid)} of {len(ALL)} evaluations valid",
        end="",
    )
    print(f"; missing {missing}" if missing else "", end="")
    print(f"; invalid {invalid}" if invalid else "")

    def score(arm: str) -> int | None:
        if arm in rows and arm not in invalid:
            return int(rows[arm]["successes"])
        return None

    def show(arm: str) -> int | None:
        s = score(arm)
        print(f"    {arm:12s} {s if s is not None else 'not read':>8}")
        return s

    print("\nnull, an unperturbed actor: 30.9 +/- 3.6, range 28-37 (E15 correction 2)")

    print("\nP1  FPO's concentration in a random subspace scores at least 20 of 50")
    p1 = [show(a) for a in ROT]
    read = [s for s in p1 if s is not None]
    p1_holds = False
    if len(read) < len(ROT):
        print("    NOT READ - an arm is missing or invalid")
    elif all(s >= 20 for s in read):
        print("    HOLDS - the subspace, not the concentration, destroys the policy")
        p1_holds = True
    elif all(s <= 5 for s in read):
        print("    FALSIFIED - the concentration alone destroys the policy")
    else:
        print("    FALSIFIED - partial, no reading")

    print("\nkeep-in-s1  FPO's input side, a random output side (not predicted)")
    s = show("keep-in-s1")
    if not p1_holds:
        print("    not read - P1 does not hold")
    elif s is None:
        print("    NOT READ - the arm is missing or invalid")
    elif s <= 10:
        print(
            "    how much each layer's output changes destroys it, whatever the direction"
        )
    elif s >= 20:
        print(
            "    FPO's output directions matter too; the same change elsewhere is absorbed"
        )
    else:
        print("    11 to 19, no reading")

    print("\nP2  FPO's update to the modulation alone scores at most 5 of 50")
    s = show("only-mod")
    if s is None:
        print("    NOT READ - the arm is missing or invalid")
    elif s <= 5:
        print("    HOLDS - the modulation alone reproduces the collapse")
    else:
        print("    FALSIFIED")

    print("\nthe other groups, alone (reported)")
    sufficient = []
    for arm in ONLY:
        s = score(arm)
        if arm != "only-mod":
            show(arm)
        if s is not None and s <= 5:
            sufficient.append(arm)
    for arm in ONLY[1:]:
        s = score(arm)
        if s is None:
            continue
        verdict = (
            "sufficient alone"
            if s <= 5
            else "not sufficient alone"
            if s >= 20
            else "partial"
        )
        print(f"    {arm:12s} {verdict}")
    if not sufficient and all(score(a) is not None for a in ONLY):
        print(
            "    no single group is sufficient: the damage needs groups in combination"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
