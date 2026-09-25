"""Do the perturbed policies fail on the same initial states the base does?

    python episode_patterns.py        # on qz103, after the evaluations

Each evaluation runs initial states 0-49 in order, so episode k is state k and
patterns compare position by position (E15 correction 2). The seven
evaluations of an unperturbed actor agree unanimously on 39 of the 50 states;
the rest are the borderline ones where the evaluation's own noise lives. A
perturbed policy that behaves like the base should disagree with that
consensus only where the base itself is unsure.
"""

import glob
import pathlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

E14 = pathlib.Path("/home/gotham/tmp/plugrl/e14")
NULL = [
    "eval-baseline",
    "eval-baseline-repeat",
    "eval-baseline-third",
    "eval-base-roundtrip",
    "eval-base-roundtrip2",
    "eval-base-roundtrip3",
    "eval-lr0-control",
]
ARMS = [
    "flip-s1",
    "flip-s2",
    "flip-s3",
    "neg",
    "flip-s1-x3",
    "flip-s1-x10",
    "flip-s1-x30",
    "neg-x3",
]


def pattern(cell: str) -> str:
    events = sorted(
        glob.glob(
            str(
                E14
                / f"runs-{cell}"
                / "ck"
                / "eval"
                / "pi0-policy"
                / cell
                / "tensorboard"
                / "events.out.tfevents.*"
            )
        )
    )
    if not events:
        return ""
    acc = EventAccumulator(events[0], size_guidance={"scalars": 0})
    acc.Reload()
    if "episode/success" not in acc.Tags()["scalars"]:
        return ""
    return "".join(
        "1" if e.value > 0.5 else "." for e in acc.Scalars("episode/success")
    )


def main() -> None:
    null = {c: pattern(c) for c in NULL}
    null = {c: p for c, p in null.items() if len(p) == 50}
    print(f"null: {len(null)} evaluations of an unperturbed actor")
    for c, p in null.items():
        print(f"  {c:24s} {p.count('1'):>2}  {p}")

    unanimous = {
        k: next(iter(null.values()))[k]
        for k in range(50)
        if len({p[k] for p in null.values()}) == 1
    }
    succeed = sum(1 for v in unanimous.values() if v == "1")
    print(
        f"unanimous states: {len(unanimous)} of 50 "
        f"({succeed} always succeed, {len(unanimous) - succeed} always fail)"
    )

    # The null's own version of the same number: each unperturbed evaluation
    # against the states the other six agree on unanimously.
    print(
        "\nleave one out: each null evaluation against the other six's unanimous states"
    )
    for c, p in null.items():
        others = [q for d, q in null.items() if d != c]
        agreed = {
            k: others[0][k] for k in range(50) if len({q[k] for q in others}) == 1
        }
        off = [k for k, v in agreed.items() if p[k] != v]
        print(f"  {c:24s} {len(off):>2} of {len(agreed)}")

    print("\narm           score  disagrees with the unanimous states")
    for arm in ARMS:
        p = pattern(f"e21-{arm}")
        if len(p) != 50:
            print(f"  {arm:12s}  not available")
            continue
        off = [k for k, v in unanimous.items() if p[k] != v]
        lost = sum(1 for k in off if unanimous[k] == "1")
        print(
            f"  {arm:12s} {p.count('1'):>3}    {len(off):>2} of {len(unanimous)}"
            f"  (lost {lost} the base always wins, won {len(off) - lost} it always loses)"
            f"  {p}"
        )


if __name__ == "__main__":
    main()
