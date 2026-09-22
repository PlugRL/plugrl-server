"""Did the trained checkpoints diverge, or are they merely bad?

Four evaluated checkpoints score 0 of 50 where the unmodified weights score 29.
That is a result about FPO only if the weights are finite. If training drove
them to NaN or Inf, or blew their scale up by orders of magnitude, then 0 of 50
is a fact about a numerical failure and saying otherwise would be dressing a
bug as a finding.

Reports, per checkpoint: how many tensors hold a non-finite value, and the
global L2 norm and max absolute value over all floating-point tensors. The
series across iterations is the informative part - a norm that climbs steadily
is a different story from one that stays flat.
"""

import sys

import torch
from safetensors import safe_open


def main():
    print("checkpoint\tnonfinite_tensors\ttensors\tglobal_l2\tmax_abs")
    for path in sys.argv[1:]:
        total_sq = 0.0
        max_abs = 0.0
        bad = 0
        n = 0
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if not t.is_floating_point():
                    continue
                n += 1
                t = t.float()
                if not torch.isfinite(t).all():
                    bad += 1
                    continue
                total_sq += float(t.pow(2).sum())
                m = float(t.abs().max())
                if m > max_abs:
                    max_abs = m
        label = path.split("/")[-2]
        print(f"{label}\t{bad}\t{n}\t{total_sq ** 0.5:.6g}\t{max_abs:.6g}")


if __name__ == "__main__":
    main()
