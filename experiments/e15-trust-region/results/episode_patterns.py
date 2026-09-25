"""Do the evaluations agree episode by episode, or only in their totals?

Three evaluations of the untrained pi0.5 with no checkpoint scored 29, 29, 29
of 50; three through the checkpoint path scored 35, 28, 29. A single GPU
forward has now been shown bit-identical before and after loading, with no
non-deterministic op on the path, so the spread is not in the model.

That leaves two readings. Either the no-checkpoint path is deterministic and
the checkpoint path is not, or every evaluation varies episode by episode and
29, 29, 29 was a coincidence of totals - about a 2% event at p=0.6, n=50.

Each evaluation runs one env in one process over initial states 0-49 in
order, so episode k is initial state k and the patterns can be compared
position by position.
"""

import glob
import pathlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

E = pathlib.Path("/home/gotham/tmp/plugrl/e14")
RUNS = [
    "eval-baseline",
    "eval-baseline-repeat",
    "eval-baseline-third",
    "eval-base-roundtrip",
    "eval-base-roundtrip2",
    "eval-base-roundtrip3",
]


def pattern(run: str) -> tuple[str, str]:
    events = sorted(
        glob.glob(
            str(
                E
                / f"runs-{run}"
                / "ck"
                / "eval"
                / "pi0-policy"
                / run
                / "tensorboard"
                / "events.out.tfevents.*"
            )
        )
    )
    if not events:
        return "", "no events"
    acc = EventAccumulator(events[0], size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    for tag in ("episode/success", "episode/reward", "episode/is_success"):
        if tag in tags:
            values = [e.value for e in acc.Scalars(tag)]
            return "".join("1" if v > 0.5 else "." for v in values), tag
    return "", "episode tags: " + ", ".join(t for t in tags if t.startswith("episode"))


def main() -> None:
    patterns = {}
    for run in RUNS:
        p, tag = pattern(run)
        patterns[run] = p
        print(f"{run:22s} {p.count('1'):>3}/{len(p):<3} {p}   [{tag}]")

    print("\npairwise: positions where two runs disagree")
    names = [r for r in RUNS if patterns[r]]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = patterns[a], patterns[b]
            n = min(len(pa), len(pb))
            diff = sum(1 for k in range(n) if pa[k] != pb[k])
            print(f"  {a:22s} vs {b:22s} {diff:>3} of {n}")


if __name__ == "__main__":
    main()
