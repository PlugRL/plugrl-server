"""Is FPO's update low-rank, and does randomising its signs destroy that?

    python delta_structure.py BASE FPO

Read after the evaluations, as mechanism, not as a registered measurement.
A gradient step is a sum of outer products over a batch, so each weight
matrix's change can be concentrated in a few directions; flipping each
element's sign at random spreads the same Frobenius norm over all of them.
A structured change of a given norm can move a layer's output far more than
an unstructured one of the same norm, which matters for reading E21: the
random arms differ from FPO's in structure as well as in direction.

For every 2-D tensor FPO changed: the share of ||Δ||² in the top 1 and top 8
singular values, for Δ and for E21's `flip-s1` sign pattern applied to it,
and the effective rank exp(entropy of the normalised squared spectrum).
"""

import collections
import math
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


def spectrum_stats(m: torch.Tensor) -> tuple[float, float, float]:
    s2 = torch.linalg.svdvals(m).pow(2)
    total = float(s2.sum())
    p = s2 / total
    entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
    return float(s2[:1].sum()) / total, float(s2[:8].sum()) / total, math.exp(entropy)


def main(base_path: str, fpo_path: str) -> None:
    torch.set_num_threads(32)
    rows = []
    g = torch.Generator().manual_seed(1)  # flip-s1's seed, same key order
    with (
        safe_open(base_path, framework="pt", device="cpu") as fa,
        safe_open(fpo_path, framework="pt", device="cpu") as fb,
    ):
        keys = []
        for key in sorted(fa.keys()):
            grp = group_of(key)
            if not grp.startswith("actor.") or grp == "actor.backbone":
                continue
            d = fb.get_tensor(key).double() - fa.get_tensor(key).double()
            if bool(d.any()):
                keys.append((key, d))
        for key, d in keys:
            r = (
                torch.randint(0, 2, d.shape, generator=g, dtype=torch.int8).double() * 2
                - 1
            )
            if d.dim() != 2 or min(d.shape) < 16:
                continue
            fpo = spectrum_stats(d.float())
            flip = spectrum_stats((r * d).float())
            rows.append((key, tuple(d.shape), fpo, flip))

    print(f"{len(rows)} two-dimensional tensors")
    print(f"{'':52s} {'top-1 share':>18s} {'top-8 share':>18s} {'effective rank':>20s}")
    print(
        f"{'':52s} {'FPO':>8s} {'flip':>9s} {'FPO':>8s} {'flip':>9s} {'FPO':>9s} {'flip':>10s}"
    )
    by_kind = collections.defaultdict(list)
    for key, shape, fpo, flip in rows:
        kind = key.split(".")[-2] if "layers" in key else key
        by_kind[kind].append((shape, fpo, flip))
    for kind, items in sorted(by_kind.items()):
        n = len(items)
        mean = [sum(it[1][i] for it in items) / n for i in range(3)]
        meanf = [sum(it[2][i] for it in items) / n for i in range(3)]
        shape = items[0][0]
        print(
            f"{kind[:34]:34s} x{n:<3d} {str(shape):>13s} "
            f"{mean[0]:8.3f} {meanf[0]:9.4f} {mean[1]:8.3f} {meanf[1]:9.4f} "
            f"{mean[2]:9.1f} {meanf[2]:10.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
