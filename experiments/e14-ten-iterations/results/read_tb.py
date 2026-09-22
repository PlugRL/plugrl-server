"""Dump the scalar series the training run logged to tensorboard.

The run was launched with --no-show-metric-table, so nothing reached the
console, but the SummaryWriter was writing all along. These are the only
per-step numbers from inside the training loop that exist: everything else
about E14's collapse has been inferred from checkpoints on disk.

Prints every scalar tag, then the first and last few values of each, so a
series that exploded, went flat, or never moved is visible without a browser.
"""

import sys

from tensorboard.backend.event_processing import event_accumulator


def main(path):
    ea = event_accumulator.EventAccumulator(
        path, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("no scalar tags in this event file")
        return

    print(f"{len(tags)} scalar tags\n")
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        vals = [e.value for e in events]
        steps = [e.step for e in events]
        lo, hi = min(vals), max(vals)
        head = ", ".join(f"{s}:{v:.6g}" for s, v in zip(steps[:4], vals[:4]))
        tail = ", ".join(f"{s}:{v:.6g}" for s, v in zip(steps[-4:], vals[-4:]))
        print(f"{tag}")
        print(f"    n={len(vals)}  min={lo:.6g}  max={hi:.6g}")
        print(f"    first  {head}")
        print(f"    last   {tail}")


if __name__ == "__main__":
    main(sys.argv[1])
