"""Print the ratio, clip, CFM-loss and advantage scalars of every tensorboard under the given roots, one line per tag: first, last, min, max."""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = ("ratio", "clip", "cfm", "loss_delta", "advantages", "kl")
for root in sys.argv[1:]:
    for ev in sorted(glob.glob(root + "/**/events.*", recursive=True)):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        print(ev)
        for t in acc.Tags()["scalars"]:
            if any(k in t for k in KEYS):
                v = [e.value for e in acc.Scalars(t)]
                print(
                    f"  {t:40s} n={len(v)} first={v[0]:.6g} last={v[-1]:.6g} "
                    f"min={min(v):.6g} max={max(v):.6g}"
                )
