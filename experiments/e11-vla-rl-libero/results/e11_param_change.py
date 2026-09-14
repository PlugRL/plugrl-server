"""Did a learn step actually change the weights?

Compares a checkpoint written by the FPO server with the base pi05_libero
weights. Most of the model is held in bfloat16, and an Adam update at a
learning rate of 1e-5 can be smaller than bfloat16's resolution, in which case
the step rounds away and the "fine-tuned" policy is the baseline.

The server's state_dict prefixes the model with "actor." and adds "critic.";
the base file has neither.

  python e11_param_change.py CHECKPOINT_DIR [BASE_WEIGHTS]
"""

import collections
import sys

import torch
from safetensors import safe_open

ckpt = sys.argv[1] + "/model.safetensors"
base = sys.argv[2] if len(sys.argv) > 2 else "/home/gotham/tmp/plugrl/ckpt/pi05_libero/model.safetensors"


def group(name):
    if name.startswith("critic."):
        return "critic"
    if "gemma_expert" in name:
        return "gemma_expert"
    if "paligemma" in name:
        return "paligemma (frozen)"
    return name.split(".")[1] if name.startswith("actor.") else name.split(".")[0]


stats = collections.defaultdict(lambda: dict(tensors=0, changed=0, params=0, changed_params=0, max_abs=0.0, dtypes=set()))
with safe_open(ckpt, "pt") as fc, safe_open(base, "pt") as fb:
    base_keys = set(fb.keys())
    for key in fc.keys():
        g = group(key)
        s = stats[g]
        t = fc.get_tensor(key)
        s["tensors"] += 1
        s["params"] += t.numel()
        s["dtypes"].add(str(t.dtype).replace("torch.", ""))
        base_key = key[len("actor."):] if key.startswith("actor.") else None
        if base_key is None or base_key not in base_keys:
            continue
        b = fb.get_tensor(base_key).to(t.dtype)
        diff = (t.float() - b.float()).abs()
        m = float(diff.max()) if diff.numel() else 0.0
        if m > 0:
            s["changed"] += 1
            s["changed_params"] += int((diff > 0).sum())
        s["max_abs"] = max(s["max_abs"], m)

print(f"{'group':<24} {'tensors':>8} {'changed':>8} {'params':>14} {'changed params':>15} {'max |delta|':>12}  dtypes")
for g, s in sorted(stats.items()):
    print(f"{g:<24} {s['tensors']:>8} {s['changed']:>8} {s['params']:>14,} {s['changed_params']:>15,} {s['max_abs']:>12.3e}  {','.join(sorted(s['dtypes']))}")
print("critic has no counterpart in the base weights; its change is not measurable this way.")
