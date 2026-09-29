"""How far did each arm's iteration move the policy? Compare checkpoints with the base.

    python movement.py BASE CHECKPOINT [CHECKPOINT ...]

E26's check_frozen.py with one line added. Per module group - `mlp`, `attn`,
`mod` and `io`, E22's partition - how many tensors moved and the relative
distance; then `expert`, the relative distance over every tensor of the
action expert (the 201 whose name contains `gemma_expert`), which is E15's
`actor.action_expert` row: 0.0151 for E14's iteration, 0.0066 for the
smallest movement E15 found destructive.
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


def is_expert(key: str) -> bool:
    return "gemma_expert" in key


def main(base_path: str, ckpts: list[str]) -> None:
    with safe_open(base_path, framework="pt", device="cpu") as fb:
        keys = [k for k in fb.keys() if group_of(k) or is_expert(k)]
        base = {k: fb.get_tensor(k) for k in keys}
    for path in ckpts:
        moved = collections.Counter()
        still = collections.Counter()
        num = collections.defaultdict(float)
        den = collections.defaultdict(float)
        with safe_open(path, framework="pt", device="cpu") as fc:
            for k in keys:
                a, b = base[k].double(), fc.get_tensor(k).double()
                d2, n2 = float((b - a).pow(2).sum()), float(a.pow(2).sum())
                labels = [g for g in (group_of(k),) if g]
                if is_expert(k):
                    labels.append("expert")
                for g in labels:
                    if torch.equal(a, b):
                        still[g] += 1
                    else:
                        moved[g] += 1
                    num[g] += d2
                    den[g] += n2
        print(path)
        for g in [name for name, _ in GROUPS] + ["expert"]:
            print(
                f"  {g:6s} moved {moved[g]:3d}  unchanged {still[g]:3d}  "
                f"relative distance {(num[g] / den[g]) ** 0.5:.6f}"
            )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
