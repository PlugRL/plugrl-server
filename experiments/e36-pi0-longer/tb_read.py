"""Print a run's training curves from its tensorboards: every iteration's value.

    python tb_read.py ROOT [ROOT ...]

For each events file under a root: the rollout success, return and length,
and the algorithm's ratio, clip, KL and CFM-loss scalars, one line per tag
with every iteration's value.
"""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = (
    "rollout/success",
    "rollout/reward",
    "rollout/length",
    "ratio_mean",
    "clipped_ratio",
    "cfm_loss_mean",
    "approx_kl",
    "clipfrac",
)
for root in sys.argv[1:]:
    for ev in sorted(glob.glob(root + "/**/events.*", recursive=True)):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        print(ev)
        for t in acc.Tags()["scalars"]:
            if any(k in t for k in KEYS):
                values = " ".join(f"{e.value:.4g}" for e in acc.Scalars(t))
                print(f"  {t:36s} {values}")
