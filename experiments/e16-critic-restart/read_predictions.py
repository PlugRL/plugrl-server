"""Evaluate E16's predictions exactly as PROTOCOL.md words them.

The point is that the verdicts are computed rather than judged. Each check
below quotes the prediction it implements, and the reading order is the one
the protocol fixes: P1, P2, P3/P4, P5.

A prediction whose inputs are incomplete prints INCONCLUSIVE rather than a
verdict, so a half-finished Phase B cannot be read as a result.

    python read_predictions.py [PHASE_A_DIR] [PHASE_B_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

from summarise import ARMS, FROM_STEP, phase_a, phase_b, value_at

SEEDS = (0, 1, 2)

# An arm runs 81,920 steps on a 4,096 buffer, so a finished one has 20
# updates. Anything short of that is still running, and its "end" would be a
# mean over fewer points than the ten the protocol's rule asks for - which is
# how a partial run gets read as a result. Seed 2 reached six updates before
# the machine ran out of memory and this printed a verdict on them.
EXPECTED_UPDATES = 20


def _line(verdict: str, text: str) -> None:
    print(f"  {verdict:<13} {text}")


def main() -> int:
    here = Path(__file__).resolve().parent
    a_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"
    b_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "results-b"

    a = phase_a(a_dir)
    b = phase_b(b_dir) if b_dir.exists() else {}

    # Phase A, per seed: the value at the checkpoint step and at the end.
    start = {s: value_at(a.get(s, []), FROM_STEP) for s in SEEDS}
    finish = {s: value_at(a.get(s, [])) for s in SEEDS}
    # Phase B, per seed and arm: the value at the end of its own run. Its
    # start is the seed's Phase A value at the checkpoint, not its own first
    # update, which is already one update of its own training.
    arm_updates = {(s, arm): len(b.get((s, arm), [])) for s in SEEDS for arm in ARMS}
    arm_end = {
        (s, arm): (
            value_at(b.get((s, arm), []))
            if arm_updates[(s, arm)] >= EXPECTED_UPDATES
            else None
        )
        for s in SEEDS
        for arm in ARMS
    }

    print("P1 - the control rises")
    print("     'the value at 409,600 exceeds the value at 327,680 on at least")
    print("      2 of 3 seeds'")
    rose = [
        s
        for s in SEEDS
        if start[s] is not None and finish[s] is not None and finish[s] > start[s]
    ]
    have = [s for s in SEEDS if start[s] is not None and finish[s] is not None]
    for s in SEEDS:
        if start[s] is None or finish[s] is None:
            _line("no data", f"seed {s}")
        else:
            _line(
                "rose" if s in rose else "fell",
                f"seed {s}: {start[s]:.1f} -> {finish[s]:.1f} "
                f"({finish[s] - start[s]:+.1f})",
            )
    if len(have) < 2:
        _line("INCONCLUSIVE", "fewer than two seeds finished Phase A")
        return 0
    p1 = len(rose) >= 2
    _line("HOLDS" if p1 else "FALSIFIED", f"{len(rose)} of {len(have)} seeds rose")
    if not p1:
        print("\nP1 failed: there is no rising stretch to read Phase B against.")
        print("PROTOCOL.md says to stop at P2 and say so.")

    print("\nP2 - resuming is faithful")
    print("     'the `all` arm's value at 409,600 lies between the minimum and")
    print("      maximum across Phase A's three seeds at that step'")
    ends = [finish[s] for s in SEEDS if finish[s] is not None]
    lo, hi = min(ends), max(ends)
    print(f"  Phase A range at the end: [{lo:.1f}, {hi:.1f}]")
    outside, seen = [], []
    for s in SEEDS:
        value = arm_end[(s, "all")]
        if value is None:
            _line(
                "incomplete",
                f"seed {s} all: {arm_updates[(s, 'all')]}/{EXPECTED_UPDATES} updates",
            )
            continue
        seen.append(s)
        inside = lo <= value <= hi
        if not inside:
            outside.append(s)
        _line("inside" if inside else "OUTSIDE", f"seed {s} all: {value:.1f}")
    if not seen:
        _line("INCONCLUSIVE", "no `all` arm has finished")
    else:
        _line(
            "HOLDS" if not outside else "FALSIFIED",
            "every `all` arm inside"
            if not outside
            else f"seeds {outside} outside the range",
        )
        if outside:
            # Stated here, and labelled, because the registered rule compares
            # a seed against the spread of all three rather than against its
            # own control. That is the weaker comparison and swapping it in
            # afterwards would be choosing the rule from the answer.
            print("  post-hoc, not the registered rule - each seed against its own:")
            for s in seen:
                own = finish[s]
                got = arm_end[(s, "all")]
                print(
                    f"    seed {s}: control {own:.1f} vs all {got:.1f} "
                    f"({100 * (got - own) / abs(own):+.1f}%)"
                )

    print("\nP3 / P4 - does an untrained value head reproduce the collapse")
    print("     P3: 'on at least 2 of 3 seeds, `except-critic` is below its own")
    print("      starting value, while `all` is not'")
    fell, rose_b, complete = [], [], []
    for s in SEEDS:
        base = start[s]
        ec, al = arm_end[(s, "except-critic")], arm_end[(s, "all")]
        if base is None or ec is None or al is None:
            _line(
                "incomplete",
                f"seed {s}: {arm_updates[(s, 'except-critic')]}/{EXPECTED_UPDATES} updates",
            )
            continue
        complete.append(s)
        if ec < base and not (al < base):
            fell.append(s)
        if ec > base:
            rose_b.append(s)
        _line(
            "FELL" if ec < base else "rose",
            f"seed {s}: start {base:.1f} -> except-critic {ec:.1f} "
            f"({ec - base:+.1f}), all {al:.1f} ({al - base:+.1f})",
        )
    if len(complete) < 2:
        _line("INCONCLUSIVE", "fewer than two seeds have all the arms they need")
    elif len(fell) >= 2:
        _line("P3 HOLDS", "E14's shape is reproduced at 272k parameters")
    elif len(rose_b) >= 2:
        _line("P4 HOLDS", "P3 is refuted - the mechanism does not do it alone")
    else:
        _line("NEITHER", "neither threshold is met on two seeds")

    print("\nP5 - which of the two fresh things carries the effect")
    print("     read only after P3 or P4; `model` differs from `except-critic`")
    print("     only in that its value head is restored too")
    for s in SEEDS:
        base = start[s]
        row = [arm_end[(s, arm)] for arm in ARMS]
        if base is None or any(v is None for v in row):
            _line("no data", f"seed {s}")
            continue
        _line(
            "",
            f"seed {s}: start {base:.1f} -> "
            + ", ".join(
                f"{arm} {v:.1f} ({v - base:+.1f})" for arm, v in zip(ARMS, row)
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
