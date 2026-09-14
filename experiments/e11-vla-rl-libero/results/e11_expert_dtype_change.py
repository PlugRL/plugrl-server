"""Which of the action expert's weights did a learn step move, by dtype?

Most of pi05_libero is held in bfloat16, whose spacing between neighbouring
values is about 0.8% of the value. An Adam step at a learning rate of 1e-5
moves a weight by about 1e-5, which is far below that spacing for a weight of
typical size, so the step may round away entirely. float32 weights have no
such floor at this scale.

Splits the action expert's tensors by dtype and by kind of layer, and reports
how many parameters changed from the base weights, the largest change, and the
median magnitude of the base weights in that group.

  python e11_expert_dtype_change.py CHECKPOINT_DIR [BASE_WEIGHTS]
"""

import collections
import sys

import torch
from safetensors import safe_open

ckpt = sys.argv[1] + "/model.safetensors"
base = sys.argv[2] if len(sys.argv) > 2 else "/home/gotham/tmp/plugrl/ckpt/pi05_libero/model.safetensors"


def kind(name):
    if "norm" in name:
        return "norm"
    if "lm_head" in name:
        return "lm_head"
    if "embed" in name:
        return "embed"
    return "linear"


rows = collections.defaultdict(lambda: dict(tensors=0, changed=0, params=0, changed_params=0, max_abs=0.0, medians=[]))
with safe_open(ckpt, "pt") as fc, safe_open(base, "pt") as fb:
    base_keys = set(fb.keys())
    for key in fc.keys():
        if not key.startswith("actor.") or "gemma_expert" not in key:
            continue
        base_key = key[len("actor."):]
        if base_key not in base_keys:
            continue
        t = fc.get_tensor(key)
        b = fb.get_tensor(base_key).to(t.dtype)
        diff = (t.float() - b.float()).abs()
        r = rows[(str(t.dtype).replace("torch.", ""), kind(key))]
        r["tensors"] += 1
        r["changed"] += int(bool((diff > 0).any()))
        r["params"] += t.numel()
        r["changed_params"] += int((diff > 0).sum())
        r["max_abs"] = max(r["max_abs"], float(diff.max()))
        r["medians"].append(float(b.float().abs().median()))

header = ("dtype", "kind", "tensors", "changed", "params", "changed_params", "fraction", "max_delta", "median_abs_w")
print("\t".join(header))
for (dtype, k), r in sorted(rows.items()):
    medians = sorted(r["medians"])
    median = medians[len(medians) // 2]
    print("\t".join(str(c) for c in (
        dtype, k, r["tensors"], r["changed"], r["params"], r["changed_params"],
        f"{r['changed_params'] / r['params']:.4f}", f"{r['max_abs']:.3e}", f"{median:.3e}")))
