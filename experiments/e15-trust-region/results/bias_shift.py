"""Did training add a systematic offset to the action output?

`actor.action_out_proj.bias` is the last thing applied before an action leaves
the policy, so a shift in it adds the same offset to every action it ever
produces. In every checkpoint compared so far it is among the tensors that
moved furthest - about 2.4% relative, whether the trust region was 0.05 or
0.002 - while the frozen backbone moved exactly 0.

A drift of that kind would explain what the hyperparameter arms could not:
every knob changes how far the weights travel, none changes the direction,
and a small constant offset on every action is enough to miss every grasp.

Prints the two bias vectors elementwise, their difference, and whether the
difference has a consistent sign - a genuine offset rather than noise.
"""

import sys

import torch
from safetensors import safe_open

KEY = "actor.action_out_proj.bias"


def load(path, key):
    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            return None
        return f.get_tensor(key).float()


def main(base_path, *others):
    base = load(base_path, KEY)
    if base is None:
        sys.exit(f"{KEY} not in {base_path}")
    print(f"{KEY}: {tuple(base.shape)}")
    print(f"base           norm={base.norm():.6g}  mean={base.mean():+.6g}  "
          f"min={base.min():+.6g}  max={base.max():+.6g}")

    for path in others:
        other = load(path, KEY)
        name = path.split("/")[-2]
        d = other - base
        same_sign = int((d > 0).sum()), int((d < 0).sum())
        print(
            f"{name:14s} norm={other.norm():.6g}  "
            f"|delta|={d.norm():.6g}  rel={d.norm() / base.norm():.6g}"
        )
        print(
            f"{'':14s} delta mean={d.mean():+.6g}  "
            f"max|delta|={d.abs().max():.6g}  "
            f"signs +{same_sign[0]}/-{same_sign[1]} of {d.numel()}"
        )

    if base.numel() <= 40:
        print("\nelementwise (base, then each delta):")
        print("  base  " + " ".join(f"{v:+.4f}" for v in base.tolist()))
        for path in others:
            other = load(path, KEY)
            d = (other - base).tolist()
            print(f"  {path.split('/')[-2][:6]:6s}" + " ".join(f"{v:+.4f}" for v in d))


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
