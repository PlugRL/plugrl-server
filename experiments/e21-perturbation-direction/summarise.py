"""Read E21's verdicts from the evaluation harness's own table.

    python summarise.py

Reads `results/stageC_eval.tsv`, copied from the cluster, and prints V4, then
P1, then P2 if P1 holds, then P3 - PROTOCOL.md's reading order. V1 to V3 are
make_arms.py's exit status, recorded in `results/construction.txt`.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TABLE = HERE / "results" / "stageC_eval.tsv"
CONSTRUCTION = HERE / "results" / "construction.txt"

RANDOM_X1 = ["flip-s1", "flip-s2", "flip-s3"]
DOSE = ["flip-s1", "flip-s1-x3", "flip-s1-x10", "flip-s1-x30"]
MIRROR = ["neg", "neg-x3"]
ALL = RANDOM_X1 + DOSE[1:] + MIRROR


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
    if "V1 V2 V3 pass" not in construction:
        print(
            "V1-V3  construction checks did not pass, or were not recorded - nothing is read"
        )
        return 1
    print("V1-V3  pass (results/construction.txt)")

    rows = read_rows()
    missing = [a for a in ALL if a not in rows]
    invalid = [
        a
        for a in ALL
        if a in rows and (rows[a]["valid"] != "true" or rows[a]["episodes"] != "50")
    ]
    print(
        f"V4     {len(ALL) - len(missing) - len(invalid)} of {len(ALL)} evaluations valid",
        end="",
    )
    print(f"; missing {missing}" if missing else "", end="")
    print(f"; invalid {invalid}" if invalid else "")

    def score(arm: str) -> int | None:
        if arm in rows and arm not in invalid:
            return int(rows[arm]["successes"])
        return None

    print("\nnull, an unperturbed actor: 30.9 +/- 3.6, range 28-37 (E15 correction 2)")

    print("\nP1  random directions at FPO's size score at least 20 of 50")
    p1 = [score(a) for a in RANDOM_X1]
    for arm, s in zip(RANDOM_X1, p1):
        print(f"    {arm:12s} {s if s is not None else 'not read':>8}")
    read = [s for s in p1 if s is not None]
    if len(read) < len(RANDOM_X1):
        print("    NOT READ - an arm is missing or invalid")
        p1_holds = False
    elif all(s >= 20 for s in read):
        print(
            "    HOLDS - B: the size of FPO's movement is not what destroys the policy"
        )
        p1_holds = True
    elif all(s <= 5 for s in read):
        print("    FALSIFIED - A: pi0.5 has no room for this movement in any direction")
        p1_holds = False
    else:
        print("    FALSIFIED - partial damage, no reading")
        p1_holds = False

    print("\nP2  FPO's update backwards scores at most 10 of 50")
    s = score("neg")
    print(f"    neg          {s if s is not None else 'not read':>8}")
    if not p1_holds:
        print("    not read - P1 does not hold")
    elif s is None:
        print("    NOT READ - the arm is missing or invalid")
    elif s <= 10:
        print("    HOLDS - B2: second order; either sign of FPO's direction destroys")
    elif s >= 20:
        print(
            "    FALSIFIED - B1: first order; the update points the wrong way specifically"
        )
    else:
        print("    FALSIFIED - 11 to 19, no reading")

    print("\nP3  dose-response, reported")
    for arm in DOSE + MIRROR:
        s = score(arm)
        print(f"    {arm:12s} {s if s is not None else 'not read':>8}")
    dose = {a: score(a) for a in DOSE}
    floor = next((a for a, s in dose.items() if s is not None and s <= 5), None)
    print(f"    smallest random multiple at 5 or less: {floor or 'none of those run'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
