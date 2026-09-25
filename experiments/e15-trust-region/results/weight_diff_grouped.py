"""Per-module distance between two checkpoints, not one number for all of them.

A global relative distance over this policy is close to meaningless. It runs
with `train_expert_only=True`, so the frozen PaliGemma backbone - the large
majority of the 821 tensors - contributes its full norm to the denominator
and none of the change to the numerator. A 0.18% global figure can hide an
expert that moved a hundred times that.

Groups by the part of the key that names a module, and reports each group's
own relative distance so the trained part is visible on its own.
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
        # gemma_expert is the trained action expert; the rest is the frozen
        # backbone. Names differ between builds, so match on substring.
        lowered = key.lower()
        if "expert" in lowered.replace("paligemma_with_expert", ""):
            return "actor.action_expert"
        return "actor.backbone"
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return key


def main(a_path: str, b_path: str):
    with safe_open(a_path, framework="pt", device="cpu") as fa:
        a_keys = set(fa.keys())
    with safe_open(b_path, framework="pt", device="cpu") as fb:
        b_keys = set(fb.keys())
    shared = sorted(a_keys & b_keys)

    num = collections.defaultdict(float)
    den = collections.defaultdict(float)
    count = collections.Counter()
    worst: dict[str, tuple[float, str]] = collections.defaultdict(lambda: (0.0, ""))

    with safe_open(a_path, framework="pt", device="cpu") as fa, \
            safe_open(b_path, framework="pt", device="cpu") as fb:
        for k in shared:
            ta, tb = fa.get_tensor(k), fb.get_tensor(k)
            if ta.shape != tb.shape or not ta.is_floating_point():
                continue
            ta, tb = ta.float(), tb.float()
            d = float((tb - ta).pow(2).sum())
            n = float(ta.pow(2).sum())
            g = group_of(k)
            num[g] += d
            den[g] += n
            count[g] += 1
            if n > 0:
                rel = (d / n) ** 0.5
                if rel > worst[g][0]:
                    worst[g] = (rel, k)

    print(f"{a_path.split('/')[-2]} -> {b_path.split('/')[-2]}")
    print(f"{'group':24s} {'tensors':>8s} {'rel dist':>12s}  largest single tensor")
    total_num = total_den = 0.0
    for g in sorted(num, key=lambda x: -(num[x] / den[x]) ** 0.5 if den[x] else 0):
        rel = (num[g] / den[g]) ** 0.5 if den[g] else float("nan")
        total_num += num[g]
        total_den += den[g]
        print(f"{g:24s} {count[g]:8d} {rel:12.6g}  {worst[g][0]:.4g} ({worst[g][1]})")
    overall = (total_num / total_den) ** 0.5 if total_den else float("nan")
    print(f"{'ALL':24s} {sum(count.values()):8d} {overall:12.6g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
