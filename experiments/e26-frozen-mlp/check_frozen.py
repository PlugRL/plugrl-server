"""Did freezing hold? Compare a trained checkpoint with the base, tensor by tensor.

    python check_frozen.py BASE CHECKPOINT [CHECKPOINT ...]

For every tensor of the action expert and its projections: whether it moved,
and the relative distance per module group - `mlp`, `attn`, `mod` and `io`,
E22's partition. A frozen arm must leave all 54 `mlp` tensors bit-identical
to the base; the control must move them.
"""

import collections
import re
import sys

import torch
from safetensors import safe_open

GROUPS = [
    (
        "mod",
        re.compile(
            r"(input_layernorm|post_attention_layernorm|model\.norm)\.dense\.(weight|bias)$"
        ),
    ),
    ("attn", re.compile(r"self_attn\.(q|k|v|o)_proj\.weight$")),
    ("mlp", re.compile(r"mlp\.(gate|up|down)_proj\.weight$")),
    (
        "io",
        re.compile(
            r"^actor\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)\.(weight|bias)$"
        ),
    ),
]


def group_of(key: str) -> str | None:
    if key.startswith("actor.paligemma_with_expert.") and "gemma_expert" not in key:
        return None
    hits = [name for name, rx in GROUPS if rx.search(key)]
    return hits[0] if len(hits) == 1 else None


def main(base_path: str, ckpts: list[str]) -> None:
    with safe_open(base_path, framework="pt", device="cpu") as fb:
        keys = [k for k in fb.keys() if group_of(k)]
        base = {k: fb.get_tensor(k) for k in keys}
    for path in ckpts:
        moved = collections.Counter()
        still = collections.Counter()
        num = collections.defaultdict(float)
        den = collections.defaultdict(float)
        with safe_open(path, framework="pt", device="cpu") as fc:
            for k in keys:
                g = group_of(k)
                a, b = base[k].double(), fc.get_tensor(k).double()
                if torch.equal(a, b):
                    still[g] += 1
                else:
                    moved[g] += 1
                num[g] += float((b - a).pow(2).sum())
                den[g] += float(a.pow(2).sum())
        print(path)
        for g, _ in GROUPS:
            print(
                f"  {g:5s} moved {moved[g]:3d}  unchanged {still[g]:3d}  "
                f"relative distance {(num[g] / den[g]) ** 0.5:.6f}"
            )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
