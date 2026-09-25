"""Build E21's checkpoints from the base policy and FPO's own first update.

    python make_arms.py BASE FPO OUT_DIR

BASE is the untrained pi0.5 in PlugRL's keys (E15's `dump_base.py`), FPO is
E14's first-iteration checkpoint. Every arm is the base with only the tensors
FPO changed replaced, by `θ + k · r ⊙ Δ` for a random ±1 pattern `r`, or by
`θ − k · Δ`. See PROTOCOL.md for why the control is a sign pattern and not a
Gaussian direction.

Exits non-zero, having printed which, unless checks V1 to V3 all pass.
"""

import collections
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# name: (kind, k, seed). kind "flip" is θ + k·r⊙Δ, "neg" is θ − k·Δ, "fpo" is
# θ + Δ and is built only to check the construction against E14's checkpoint.
ARMS = {
    "fpo": ("fpo", 1, None),
    "flip-s1": ("flip", 1, 1),
    "flip-s2": ("flip", 1, 2),
    "flip-s3": ("flip", 1, 3),
    "neg": ("neg", 1, None),
    "flip-s1-x3": ("flip", 3, 1),
    "flip-s1-x10": ("flip", 10, 1),
    "flip-s1-x30": ("flip", 30, 1),
    "neg-x3": ("neg", 3, None),
}

FPO_EXPERT_DISTANCE = 0.0151  # E15, weight-distances.txt, grouped_lr0_vs_iter01
V3_TOLERANCE = 0.05


def group_of(key: str) -> str:
    # E15's weight_diff_grouped.py, unchanged, so the distance is its measure.
    parts = key.split(".")
    if key.startswith("critic"):
        return "critic"
    if "paligemma_with_expert" in key:
        lowered = key.lower()
        if "expert" in lowered.replace("paligemma_with_expert", ""):
            return "actor.action_expert"
        return "actor.backbone"
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return key


def build(kind: str, k: int, seed: int | None, base: dict, delta: dict) -> dict:
    out = dict(base)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    for key in sorted(delta):
        # float64 throughout: in float32, b − a is exact only where b/a lies in
        # [1/2, 2], and elsewhere θ + Δ lands a step away from b (amendment 1).
        theta = base[key].double()
        d = delta[key]
        if kind == "fpo":
            new = theta + d
        elif kind == "neg":
            new = theta - k * d
        else:
            r = (
                torch.randint(0, 2, d.shape, generator=g, dtype=torch.int8).double() * 2
                - 1
            )
            new = theta + k * r * d
        out[key] = new.to(base[key].dtype).contiguous()
    return out


def expert_distance(arm: dict, base: dict) -> dict[str, float]:
    num = collections.defaultdict(float)
    den = collections.defaultdict(float)
    for key, theta in base.items():
        g = group_of(key)
        if not g.startswith("actor.") or g == "actor.backbone":
            continue
        t = theta.float()
        num[g] += float((arm[key].float() - t).pow(2).sum())
        den[g] += float(t.pow(2).sum())
    return {g: (num[g] / den[g]) ** 0.5 for g in num}


def main(base_path: str, fpo_path: str, out_dir: str) -> int:
    out_root = pathlib.Path(out_dir)
    failures = []

    with safe_open(base_path, framework="pt", device="cpu") as fh:
        base = {key: fh.get_tensor(key) for key in fh.keys()}
    print(f"base: {len(base)} tensors")

    delta, fpo_changed = {}, {}
    with safe_open(fpo_path, framework="pt", device="cpu") as fh:
        for key in sorted(fh.keys()):
            g = group_of(key)
            if not g.startswith("actor.") or g == "actor.backbone":
                continue
            b = fh.get_tensor(key)
            d = b.double() - base[key].double()
            if bool(d.any()):
                delta[key] = d
                fpo_changed[key] = b
    print(f"FPO changed {len(delta)} actor tensors")
    untouched = sorted(set(base) - set(delta))
    print(f"left as the base: {len(untouched)} tensors")

    print(
        f"\n{'arm':14s} {'k':>3s} {'expert dist':>12s} {'target':>9s} {'ratio':>7s}  other groups"
    )
    for name, (kind, k, seed) in ARMS.items():
        arm = build(kind, k, seed, base, delta)

        # V1: the construction reproduces FPO's own actor, bit for bit.
        if kind == "fpo":
            bad = [key for key in delta if not torch.equal(arm[key], fpo_changed[key])]
            if bad:
                failures.append(
                    f"V1 fpo: {len(bad)} tensors differ from E14's, e.g. {bad[0]}"
                )

        dist = expert_distance(arm, base)
        target = k * FPO_EXPERT_DISTANCE
        ratio = dist["actor.action_expert"] / target
        others = "  ".join(
            f"{g.split('.')[1]} {v:.4f}"
            for g, v in sorted(dist.items())
            if g != "actor.action_expert"
        )
        print(
            f"{name:14s} {k:3d} {dist['actor.action_expert']:12.6f} {target:9.4f} {ratio:7.4f}  {others}"
        )
        # V3: the arm moved as far as it was built to, after rounding.
        if abs(ratio - 1) > V3_TOLERANCE:
            failures.append(f"V3 {name}: expert distance {ratio:.4f} of target")

        path = out_root / name
        path.mkdir(parents=True, exist_ok=True)
        save_file(arm, str(path / "model.safetensors"))
        del arm

        # V2: read the saved file back; everything FPO did not change is the base's.
        with safe_open(
            str(path / "model.safetensors"), framework="pt", device="cpu"
        ) as fh:
            keys = set(fh.keys())
            if keys != set(base):
                failures.append(f"V2 {name}: key set differs from the base")
            for key in untouched:
                if key not in keys:
                    continue
                t = fh.get_tensor(key)
                if (
                    t.dtype != base[key].dtype
                    or t.shape != base[key].shape
                    or not torch.equal(t, base[key])
                ):
                    failures.append(f"V2 {name}: {key} differs from the base")
                    break
            for key in delta:
                t = fh.get_tensor(key)
                if t.dtype != base[key].dtype or t.shape != base[key].shape:
                    failures.append(f"V2 {name}: {key} changed dtype or shape")
                    break

    print()
    if failures:
        for f in failures:
            print("FAIL", f)
        return 1
    print(
        "V1 V2 V3 pass: fpo reproduces E14's actor, every other tensor is the base's,"
    )
    print(f"and every arm's expert distance is within {V3_TOLERANCE:.0%} of its target")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
