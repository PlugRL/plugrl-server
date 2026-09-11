"""E10: what one action inference costs, per experiments/e10-vla-forward-cost/PROTOCOL.md.

One process launch measures one cell. The protocol asks for three independent
launches per cell rather than three loops inside one, because a single process
measures one allocator state and one set of resident kernels.

Weights are random. The protocol committed to that in advance, on the grounds
that weights do not change what a forward costs - which is fortunate, since
the checkpoints live behind a host this cluster cannot resolve.
"""

import argparse
import json
import statistics
import sys
import time


def build_observation(cfg, batch, device, torch):
    """Build the model's own declared input, as torch tensors on `device`.

    Taken from `inputs_spec` rather than written by hand, so the shapes are
    the model's opinion and not mine.
    """
    from openpi.models import model as _model

    obs_spec, act_spec = cfg.inputs_spec(batch_size=batch)

    def leaf(spec):
        name = str(spec.dtype)
        if "bool" in name:
            return torch.ones(tuple(spec.shape), dtype=torch.bool, device=device)
        if "int" in name or "uint" in name:
            return torch.zeros(tuple(spec.shape), dtype=torch.long, device=device)
        return torch.zeros(tuple(spec.shape), dtype=torch.float32, device=device)

    # `inputs_spec` declares images channels-last, which is the JAX model's
    # layout. The PyTorch path rejects it: preprocess_observation_pytorch only
    # converts back to channels-first for input that already was, so a
    # channels-last tensor reaches the vision tower's conv2d untouched and it
    # reads 224 as the channel count. Feed it channels-first and say so.
    images = {}
    for k, v in obs_spec.images.items():
        t = leaf(v)
        if t.ndim == 4 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2).contiguous()
        images[k] = t
    image_masks = {k: leaf(v) for k, v in obs_spec.image_masks.items()}
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=leaf(obs_spec.state),
        tokenized_prompt=(
            leaf(obs_spec.tokenized_prompt)
            if obs_spec.tokenized_prompt is not None
            else None
        ),
        tokenized_prompt_mask=(
            leaf(obs_spec.tokenized_prompt_mask)
            if obs_spec.tokenized_prompt_mask is not None
            else None
        ),
    )
    shapes = {
        "images": {k: list(v.shape) for k, v in images.items()},
        "state": list(obs.state.shape),
        "tokens": (
            list(obs.tokenized_prompt.shape)
            if obs.tokenized_prompt is not None
            else None
        ),
    }
    return obs, shapes, act_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny", choices=["tiny", "base"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--denoising-steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="as plugrl's wrapper does")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from openpi.models import pi0_config
    from openpi.models_pytorch import pi0_pytorch

    if a.variant == "tiny":
        # Exactly what plugrl-server's pi0-policy defaults to, via
        # TrainConfig "pi05_tiny_libero".
        cfg = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_tiny",
            action_expert_variant="gemma_expert_tiny",
            discrete_state_input=False,
        )
    else:
        cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10)

    device = a.device
    model = pi0_pytorch.PI0Pytorch(config=cfg)
    model = model.to(device)
    model.eval()
    if a.bf16:
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    try:
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    except AttributeError:
        pass

    params = sum(p.numel() for p in model.parameters())
    obs, shapes, _ = build_observation(cfg, a.batch, device, torch)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def one():
        with torch.inference_mode():
            return model.sample_actions(device, obs, num_steps=a.denoising_steps)

    valid = True
    action_shape = None
    try:
        # PI0Pytorch wraps sample_actions in torch.compile(mode="max-autotune"),
        # so the first call is not a forward pass, it is a compilation. Timed
        # separately because it is an operational cost in its own right: a
        # server answering its first infer blocks for it, and the protocol's
        # own liveness timers are measured in seconds.
        t_first = time.perf_counter()
        out = one()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - t_first) * 1000.0

        for _ in range(max(0, a.warmup - 1)):
            out = one()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        action_shape = list(out.shape)

        times = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            one()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - a blocked path is a result
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # The protocol's validity rule: a run counts only if the action came back
    # with the shape the config declares.
    expect = [a.batch, cfg.action_horizon, cfg.action_dim]
    if action_shape != expect:
        valid = False
        print(f"action shape {action_shape} != expected {expect}", file=sys.stderr)

    times.sort()
    peak = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.startswith("cuda")
        else 0.0
    )
    row = [
        f"pi0-{a.variant}",
        device,
        "bf16-mixed" if a.bf16 else "fp32",
        params,
        a.batch,
        a.denoising_steps,
        a.launch,
        a.warmup,
        a.n,
        round(statistics.fmean(times), 3),
        round(times[len(times) // 2], 3),
        round(times[int(len(times) * 0.95) - 1], 3),
        # torch.compile recompiles more than once, and a recompile that lands
        # inside the measured window is a 9 s call sitting next to 40 ms ones.
        # The max makes that visible instead of letting the mean absorb it.
        round(times[-1], 3),
        round(peak, 1),
        round(first_call_ms, 1),
        json.dumps(action_shape),
        str(valid).lower(),
    ]
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write("\t".join(str(c) for c in row) + "\n")
    print("\t".join(str(c) for c in row))
    print("input shapes:", json.dumps(shapes), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
