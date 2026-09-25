"""Why did V1 fail - and does float64 arithmetic make θ + Δ reproduce FPO?

    python diagnose_v1.py BASE FPO

The first build computed Δ = b − a and θ + Δ in float32. A float32 difference
is exact when b and a are within a factor of two of each other (Sterbenz), so
the suspicion is elements near zero or crossing it, where a + (b − a) can land
one step away from b. This counts, per stored dtype, the tensors and elements
where the float32 path misses, how many of those elements sit where b/a is
outside [1/2, 2], and whether the float64 path misses anywhere at all.
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


def main(base_path: str, fpo_path: str) -> None:
    tensors = collections.Counter()
    miss32_tensors = collections.Counter()
    miss32_elements = collections.Counter()
    outside = collections.Counter()
    miss64_tensors = collections.Counter()
    total_elements = collections.Counter()
    worst = (0, "")
    with (
        safe_open(base_path, framework="pt", device="cpu") as fa,
        safe_open(fpo_path, framework="pt", device="cpu") as fb,
    ):
        for key in sorted(fa.keys()):
            g = group_of(key)
            if not g.startswith("actor.") or g == "actor.backbone":
                continue
            a, b = fa.get_tensor(key), fb.get_tensor(key)
            if not bool((b.float() != a.float()).any()):
                continue
            dt = str(a.dtype)
            tensors[dt] += 1
            total_elements[dt] += a.numel()

            d32 = b.float() - a.float()
            r32 = (a.float() + d32).to(a.dtype)
            m = r32 != b
            if bool(m.any()):
                miss32_tensors[dt] += 1
                n = int(m.sum())
                miss32_elements[dt] += n
                af, bf = a.float()[m], b.float()[m]
                ratio = bf / af
                outside[dt] += int(
                    ((ratio < 0.5) | (ratio > 2) | ~torch.isfinite(ratio)).sum()
                )
                if n > worst[0]:
                    worst = (n, key)

            d64 = b.double() - a.double()
            r64 = (a.double() + d64).to(a.dtype)
            if not torch.equal(r64, b):
                miss64_tensors[dt] += 1

    for dt in tensors:
        print(
            f"{dt:15s} changed {tensors[dt]:4d} tensors, {total_elements[dt]:>11d} elements | "
            f"float32 path misses {miss32_tensors[dt]:3d} tensors, {miss32_elements[dt]:>8d} elements, "
            f"{outside[dt]:>8d} of them with b/a outside [1/2, 2] | "
            f"float64 path misses {miss64_tensors[dt]} tensors"
        )
    print(f"most misses in one tensor: {worst[0]} in {worst[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
