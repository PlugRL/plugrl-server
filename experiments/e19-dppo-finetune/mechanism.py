"""What did DPPO do to the restored policies, beyond the return?

    python mechanism.py

Not a registered measurement; read after the verdicts, as E18's mechanism
section was. Per seed:

* the actor's relative movement, ||θ_end − θ_start|| / ||θ_start||, over the
  actor's tensors only - E15's measure, and E20's for FPO on these same
  checkpoints;
* `losses/approx_kl` and `losses/clipfrac` over the twenty updates;
* the return curve, so `start` - a single iteration - can be seen against its
  neighbours.
"""

import pathlib

import torch
from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
PHASE_A = HERE.parent / "e16-critic-restart" / "results" / "fpo" / "fpo-policy"
RUNS = HERE / "results" / "dppo" / "fpo-policy"
FROM_STEP = 327680


def last_numbered(root: pathlib.Path, near: int | None = None) -> pathlib.Path:
    steps = [int(p.name) for p in root.iterdir() if p.name.isdigit()]
    step = max(steps) if near is None else min(steps, key=lambda s: abs(s - near))
    return root / str(step) / "model.safetensors"


def actor(path: pathlib.Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        return {
            k: fh.get_tensor(k).float() for k in fh.keys() if k.startswith("actor.")
        }


def scalars(run: pathlib.Path, tag: str) -> list[float]:
    events = sorted((run / "tensorboard").glob("events.out.tfevents.*"))
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    return [e.value for e in acc.Scalars(tag)]


def main() -> None:
    for seed in (0, 1, 2):
        run = RUNS / f"halfcheetah-seed{seed}"
        a = actor(last_numbered(PHASE_A / f"halfcheetah-seed{seed}", FROM_STEP))
        b = actor(last_numbered(run))
        num = sum(float((b[k] - a[k]).pow(2).sum()) for k in a)
        den = sum(float(a[k].pow(2).sum()) for k in a)
        kl = scalars(run, "losses/approx_kl")
        clip = scalars(run, "losses/clipfrac")
        reward = scalars(run, "rollout/reward")
        print(f"seed {seed}")
        print(f"  actor movement      {(num / den) ** 0.5:.5f}")
        if kl:
            print(
                f"  approx_kl           median {sorted(kl)[len(kl) // 2]:.2e}  max {max(kl):.2e}  ({len(kl)} values)"
            )
        if clip:
            print(f"  clipfrac            max {max(clip):.4f}")
        if reward:
            print("  rollout/reward      " + " ".join(f"{r:.0f}" for r in reward))
            if len(reward) >= 20:
                first3 = sum(reward[:3]) / 3
                last10 = sum(reward[10:20]) / 10
                print(f"  first 3 mean {first3:.1f}   last 10 mean {last10:.1f}")


if __name__ == "__main__":
    main()
