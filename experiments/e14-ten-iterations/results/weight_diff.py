"""How far did the weights actually move?

A previous check compared global L2 NORMS across checkpoints and found them
equal to eight parts per million. That is a weak diagnostic and it was
over-read: weights can move a long way while keeping their norm, and an equal
norm is not an unchanged tensor. The quantity that answers the question is the
norm of the DIFFERENCE.

Prints, for each pair, the relative distance ||b - a|| / ||a|| over the tensors
they share, the largest single-tensor relative distance and which tensor it
is, and how many tensors are missing from either side or differ in shape -
because two checkpoints of different file sizes may not hold the same tensors
at all, and a comparison that silently skips the interesting half is worse
than none.
"""

import sys

import torch
from safetensors import safe_open


def load_keys(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return set(f.keys())


def compare(a_path, b_path):
    a_keys, b_keys = load_keys(a_path), load_keys(b_path)
    only_a, only_b = a_keys - b_keys, b_keys - a_keys
    shared = sorted(a_keys & b_keys)

    num_sq = 0.0
    den_sq = 0.0
    worst = (0.0, "")
    shape_mismatch = 0
    compared = 0

    with safe_open(a_path, framework="pt", device="cpu") as fa, \
            safe_open(b_path, framework="pt", device="cpu") as fb:
        for k in shared:
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            if ta.shape != tb.shape:
                shape_mismatch += 1
                continue
            if not ta.is_floating_point():
                continue
            ta = ta.float()
            tb = tb.float()
            d = float((tb - ta).pow(2).sum())
            n = float(ta.pow(2).sum())
            num_sq += d
            den_sq += n
            compared += 1
            if n > 0:
                rel = (d / n) ** 0.5
                if rel > worst[0]:
                    worst = (rel, k)

    rel_total = (num_sq / den_sq) ** 0.5 if den_sq > 0 else float("nan")
    print(f"{a_path.split('/')[-2]} -> {b_path.split('/')[-2]}")
    print(f"  relative distance ||b-a||/||a||   {rel_total:.6g}")
    print(f"  largest per-tensor relative       {worst[0]:.6g}  ({worst[1]})")
    print(f"  tensors compared                  {compared}")
    print(f"  only in a / only in b / reshaped  {len(only_a)} / {len(only_b)} / {shape_mismatch}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
