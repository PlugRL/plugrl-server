"""Which of FPO's 208 changed tensors fall in which module group, and their dtypes."""

import collections
import re
import sys

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


def main(base_path: str, fpo_path: str) -> None:
    counts = collections.Counter()
    dtypes = collections.defaultdict(collections.Counter)
    dims = collections.defaultdict(collections.Counter)
    unmatched, unchanged = [], []
    with safe_open(base_path, "pt") as fa, safe_open(fpo_path, "pt") as fb:
        for key in sorted(fa.keys()):
            if not key.startswith("actor.") or (
                "paligemma_with_expert" in key and "gemma_expert" not in key
            ):
                continue
            a, b = fa.get_tensor(key), fb.get_tensor(key)
            if not bool((a.double() != b.double()).any()):
                if "gemma_expert" in key or key.split(".")[1] in (
                    "action_in_proj",
                    "action_out_proj",
                    "time_mlp_in",
                    "time_mlp_out",
                ):
                    unchanged.append(key)
                continue
            hits = [name for name, rx in GROUPS if rx.search(key)]
            if len(hits) != 1:
                unmatched.append((key, hits))
                continue
            counts[hits[0]] += 1
            dtypes[hits[0]][str(a.dtype)] += 1
            dims[hits[0]][a.dim()] += 1
    for name, _ in GROUPS:
        print(
            f"{name:5s} {counts[name]:4d} tensors  dtypes {dict(dtypes[name])}  dims {dict(dims[name])}"
        )
    print(f"total {sum(counts.values())}")
    print(f"changed but unmatched or matched twice: {unmatched}")
    print(f"unchanged trainable tensors: {unchanged}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
