"""What does FPO's first-iteration update look like, element by element?

    python probe_delta.py BASE FPO

Before choosing a random control for it: which tensors it changed, in which
dtype they are stored, and - for the bf16 ones - how many representable steps
each element moved. A Gaussian perturbation scaled to the same norm would be
rounded on saving; if most of the update is one or two bf16 steps per element,
that rounding would change the perturbation's size and shape.

Output from the run that shaped E21's design: `results/probe-delta.txt`.
"""

import collections
import sys

import torch
from safetensors import safe_open


def group_of(key: str) -> str:
    parts = key.split(".")
    if key.startswith("critic"):
        return "critic"
    if "paligemma_with_expert" in key:
        lowered = key.lower()
        if "expert" in lowered.replace("paligemma_with_expert", ""):
            return "actor.action_expert"
        return "actor.backbone"
    return ".".join(parts[:2]) if len(parts) >= 2 else key


def ulp_bf16(w: torch.Tensor) -> torch.Tensor:
    # Spacing between adjacent bf16 values at w: 2^(exponent - 7).
    a = w.abs().float().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def main(base_path: str, fpo_path: str) -> None:
    changed = collections.Counter()
    unchanged = collections.Counter()
    dtypes = collections.defaultdict(collections.Counter)
    steps = collections.Counter()
    frac_nonzero = []
    with (
        safe_open(base_path, framework="pt", device="cpu") as fa,
        safe_open(fpo_path, framework="pt", device="cpu") as fb,
    ):
        for k in sorted(fa.keys()):
            g = group_of(k)
            if not g.startswith("actor.") or g == "actor.backbone":
                continue
            a, b = fa.get_tensor(k), fb.get_tensor(k)
            d = b.float() - a.float()
            if not bool(d.any()):
                unchanged[g] += 1
                continue
            changed[g] += 1
            dtypes[g][str(a.dtype)] += 1
            frac_nonzero.append(
                (float((d != 0).float().mean()), k, str(a.dtype), a.numel())
            )
            if a.dtype == torch.bfloat16:
                u = (d.abs() / ulp_bf16(a)).round().clamp_max(10).to(torch.int64)
                for v, n in zip(*torch.unique(u, return_counts=True)):
                    steps[int(v)] += int(n)

    print("changed tensors per group:", dict(changed))
    print("unchanged tensors per group:", dict(unchanged))
    for g, c in dtypes.items():
        print(f"  {g}: {dict(c)}")
    total = sum(steps.values())
    if total:
        print("bf16 elements, |delta| in bf16 steps (10 = 10 or more):")
        for v in sorted(steps):
            print(f"  {v:>2}: {steps[v] / total:.4f}")
    frac_nonzero.sort()
    print("fraction of elements changed, lowest and highest five tensors:")
    for row in frac_nonzero[:5] + frac_nonzero[-5:]:
        print(f"  {row[0]:.4f}  {row[2]:15s} {row[3]:>10d}  {row[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
