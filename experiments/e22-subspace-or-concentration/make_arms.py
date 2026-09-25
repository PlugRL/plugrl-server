"""Build E22's checkpoints from the base policy and FPO's first update.

    python make_arms.py BASE FPO OUT_DIR

Every arm is the base with some of the 208 tensors FPO changed replaced:

* `rot-sN`     every 2-D Δ = U S Vᵀ becomes U' S V'ᵀ, U' and V' Haar-random;
               every 1-D Δ becomes r ⊙ Δ
* `keep-in-s1` U' S Vᵀ - FPO's input side, rot-s1's output side; 1-D as rot
* `only-<g>`   θ + Δ on module group g, the base elsewhere
* `fpo`        θ + Δ on all four groups, built only as a check

See PROTOCOL.md. Exits non-zero, having printed which, unless V1 to V4 pass.
"""

import collections
import pathlib
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
GROUP_SIZES = {"mod": 74, "attn": 72, "mlp": 54, "io": 8}

# name: (kind, seed or group)
ARMS = {
    "fpo": ("fpo", None),
    "rot-s1": ("rot", 1),
    "rot-s2": ("rot", 2),
    "rot-s3": ("rot", 3),
    "keep-in-s1": ("keep-in", 1),
    "only-mod": ("only", "mod"),
    "only-attn": ("only", "attn"),
    "only-mlp": ("only", "mlp"),
    "only-io": ("only", "io"),
}

FPO_EXPERT_DISTANCE = 0.0151  # E15, weight-distances.txt
V3_TOLERANCE = 0.05
V4_TOLERANCE = 0.05


def e15_group(key: str) -> str:
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


def module_group(key: str) -> str | None:
    hits = [name for name, rx in GROUPS if rx.search(key)]
    return hits[0] if len(hits) == 1 else None


def haar(rows: int, cols: int, g: torch.Generator) -> torch.Tensor:
    """A rows x cols matrix with Haar-random orthonormal columns."""
    z = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def top8_share(m: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(m.float()).double().pow(2)
    return float(s2[:8].sum() / s2.sum())


def build(kind, arg, base, delta, groups, svd) -> dict:
    out = dict(base)
    if kind in ("fpo", "only"):
        for key in sorted(delta):
            if kind == "only" and groups[key] != arg:
                continue
            new = base[key].double() + delta[key]
            out[key] = new.to(base[key].dtype).contiguous()
        return out

    g = torch.Generator().manual_seed(arg)  # U', then V', per 2-D tensor
    g_sign = torch.Generator().manual_seed(arg)  # r, per 1-D tensor
    for key in sorted(delta):
        d = delta[key]
        if d.dim() == 2:
            u, s, vh = svd[key]
            u_rand = haar(u.shape[0], u.shape[1], g)
            v_rand = haar(
                vh.shape[1], vh.shape[0], g
            )  # drawn by keep-in too, so U' matches
            if kind == "rot":
                new_d = u_rand @ torch.diag(s) @ v_rand.T
            else:
                new_d = u_rand @ torch.diag(s) @ vh
        else:
            r = (
                torch.randint(
                    0, 2, d.shape, generator=g_sign, dtype=torch.int8
                ).double()
                * 2
                - 1
            )
            new_d = r * d
        out[key] = (base[key].double() + new_d).to(base[key].dtype).contiguous()
    return out


def distances(arm: dict, base: dict) -> dict[str, float]:
    num = collections.defaultdict(float)
    den = collections.defaultdict(float)
    for key, theta in base.items():
        g = e15_group(key)
        if not g.startswith("actor.") or g == "actor.backbone":
            continue
        t = theta.double()
        num[g] += float((arm[key].double() - t).pow(2).sum())
        den[g] += float(t.pow(2).sum())
    return {g: (num[g] / den[g]) ** 0.5 for g in num}


def main(base_path: str, fpo_path: str, out_dir: str) -> int:
    torch.set_num_threads(64)
    out_root = pathlib.Path(out_dir)
    failures = []

    with safe_open(base_path, framework="pt", device="cpu") as fh:
        base = {key: fh.get_tensor(key) for key in fh.keys()}
    print(f"base: {len(base)} tensors")

    delta, fpo_changed = {}, {}
    with safe_open(fpo_path, framework="pt", device="cpu") as fh:
        for key in sorted(fh.keys()):
            g = e15_group(key)
            if not g.startswith("actor.") or g == "actor.backbone":
                continue
            b = fh.get_tensor(key)
            d = b.double() - base[key].double()
            if bool(d.any()):
                delta[key] = d
                fpo_changed[key] = b
    untouched = sorted(set(base) - set(delta))
    print(f"FPO changed {len(delta)} actor tensors; {len(untouched)} left as the base")

    # V1, first half: the four groups are disjoint and cover what FPO changed.
    groups = {key: module_group(key) for key in delta}
    counts = collections.Counter(groups.values())
    if None in counts or dict(counts) != GROUP_SIZES:
        failures.append(f"V1 groups: {dict(counts)} against {GROUP_SIZES}")
    print(f"groups: {dict(counts)}")

    svd = {}
    fpo_top8 = collections.defaultdict(list)
    for key, d in delta.items():
        if d.dim() == 2:
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            svd[key] = (u, s, vh)
            if groups[key] in ("mod", "attn", "mlp"):
                s2 = s.pow(2)
                fpo_top8[groups[key]].append(float(s2[:8].sum() / s2.sum()))
    fpo_top8 = {g: sum(v) / len(v) for g, v in fpo_top8.items()}
    print(
        f"SVD of {len(svd)} 2-D tensors; FPO's mean top-8 share: "
        + "  ".join(f"{g} {v:.3f}" for g, v in sorted(fpo_top8.items()))
    )

    print(
        f"\n{'arm':12s} {'expert':>9s}  other groups   | top-8 share of the realised perturbation"
    )
    for name, (kind, arg) in ARMS.items():
        arm = build(kind, arg, base, delta, groups, svd)
        changed = [k for k in delta if not torch.equal(arm[k], base[k])]

        if kind == "fpo":  # V1, second half
            bad = [k for k in delta if not torch.equal(arm[k], fpo_changed[k])]
            if bad:
                failures.append(
                    f"V1 fpo: {len(bad)} tensors differ from E14's, e.g. {bad[0]}"
                )
        if kind == "only":  # V3 for the localisation arms
            want = [k for k in delta if groups[k] == arg]
            bad = [k for k in want if not torch.equal(arm[k], fpo_changed[k])]
            extra = [
                k
                for k in delta
                if groups[k] != arg and not torch.equal(arm[k], base[k])
            ]
            if bad or extra:
                failures.append(
                    f"V3 {name}: {len(bad)} group tensors differ from E14's, {len(extra)} outside the group changed"
                )

        dist = distances(arm, base)
        expert = dist["actor.action_expert"]
        others = " ".join(
            f"{g.split('.')[1]} {v:.4f}"
            for g, v in sorted(dist.items())
            if g != "actor.action_expert"
        )
        structure = ""
        if kind in ("rot", "keep-in"):  # V3 and V4
            if abs(expert / FPO_EXPERT_DISTANCE - 1) > V3_TOLERANCE:
                failures.append(f"V3 {name}: expert distance {expert:.5f}")
            shares = collections.defaultdict(list)
            for k in changed:
                if arm[k].dim() == 2 and groups[k] in ("mod", "attn", "mlp"):
                    shares[groups[k]].append(
                        top8_share(arm[k].double() - base[k].double())
                    )
            realised = {g: sum(v) / len(v) for g, v in shares.items()}
            for g, v in realised.items():
                if abs(v - fpo_top8[g]) > V4_TOLERANCE:
                    failures.append(
                        f"V4 {name}: {g} top-8 share {v:.3f} against FPO's {fpo_top8[g]:.3f}"
                    )
            structure = "  ".join(f"{g} {v:.3f}" for g, v in sorted(realised.items()))
        print(f"{name:12s} {expert:9.6f}  {others} | {structure}")

        path = out_root / name
        path.mkdir(parents=True, exist_ok=True)
        save_file(arm, str(path / "model.safetensors"))
        del arm

        # V2: read the saved file back; everything the arm did not change is the base's.
        keep = set(untouched) | {k for k in delta if k not in changed}
        with safe_open(
            str(path / "model.safetensors"), framework="pt", device="cpu"
        ) as fh:
            if set(fh.keys()) != set(base):
                failures.append(f"V2 {name}: key set differs from the base")
            for key in sorted(keep):
                t = fh.get_tensor(key)
                if (
                    t.dtype != base[key].dtype
                    or t.shape != base[key].shape
                    or not torch.equal(t, base[key])
                ):
                    failures.append(f"V2 {name}: {key} differs from the base")
                    break
            for key in changed:
                t = fh.get_tensor(key)
                if t.dtype != base[key].dtype or t.shape != base[key].shape:
                    failures.append(f"V2 {name}: {key} changed dtype or shape")
                    break

    print()
    if failures:
        for f in failures:
            print("FAIL", f)
        return 1
    print("V1 V2 V3 V4 pass: fpo reproduces E14's actor from four disjoint groups,")
    print(
        "every unchanged tensor is the base's, every rotation arm moved FPO's distance"
    )
    print(
        f"within {V3_TOLERANCE:.0%} and kept FPO's concentration within {V4_TOLERANCE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
