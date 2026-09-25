"""How far did each E20 run move its actor, in E15's measure?

    python actor_movement.py

Relative distance ||θ_end − θ_start|| / ||θ_start|| over the actor's tensors
only - not the critic, which every arm replaced with a random one, and not the
observation statistics, which are buffers. `start` is the Phase A checkpoint
the run restored from, the one nearest step 327,680 for its seed; `end` is the
run's own last checkpoint.

The number exists to be set beside pi0.5's: E15 measured 0.0066 to 0.0151 of
relative movement in pi0.5's action expert, and every one of those policies
scored zero.
"""

import pathlib

import torch
from safetensors import safe_open

HERE = pathlib.Path(__file__).resolve().parent
PHASE_A = HERE.parent / "e16-critic-restart" / "results" / "fpo" / "fpo-policy"
RUNS = HERE / "results" / "fpo" / "fpo-policy"
FROM_STEP = 327680


def start_checkpoint(seed: int) -> pathlib.Path:
    root = PHASE_A / f"halfcheetah-seed{seed}"
    steps = [int(p.name) for p in root.iterdir() if p.name.isdigit()]
    nearest = min(steps, key=lambda s: abs(s - FROM_STEP))
    return root / str(nearest) / "model.safetensors"


def end_checkpoint(arm: str, seed: int) -> pathlib.Path:
    # The last save lands on 81,920 or 81,921, as E16's did on 327,680 or
    # 327,681; take whichever the run wrote.
    root = RUNS / f"{arm}-seed{seed}"
    last = max(int(p.name) for p in root.iterdir() if p.name.isdigit())
    return root / str(last) / "model.safetensors"


def actor(path: pathlib.Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        return {
            k: fh.get_tensor(k).float() for k in fh.keys() if k.startswith("actor.")
        }


def main() -> None:
    print("arm     seed  start step  end step  relative actor movement")
    for arm in ("reward", "zero"):
        for seed in (0, 1, 2):
            start_path = start_checkpoint(seed)
            end_path = end_checkpoint(arm, seed)
            a, b = actor(start_path), actor(end_path)
            num = sum(float((b[k] - a[k]).pow(2).sum()) for k in a)
            den = sum(float(a[k].pow(2).sum()) for k in a)
            print(
                f"{arm:7s} {seed:4d}  {start_path.parent.name:>10s}  "
                f"{end_path.parent.name:>8s}  {(num / den) ** 0.5:.4f}"
            )


if __name__ == "__main__":
    main()
