"""E46 exploration: the server's own timing scalars for each explore_alone.sh run.

python explore_timing.py OUT_DIR
"""

import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = pathlib.Path(sys.argv[1])
for run in sorted((out / "fpo" / "fpo-policy").iterdir()):
    events = sorted((run / "tensorboard").glob("events.*"))
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    tags = [
        t
        for t in acc.Tags()["scalars"]
        if any(k in t for k in ("time", "duration", "_ms", "_s", "fps", "latency"))
    ]
    print(run.name)
    for tag in tags:
        values = [e.value for e in acc.Scalars(tag)]
        print(f"   {tag:<50} " + " ".join(f"{v:.5g}" for v in values))
