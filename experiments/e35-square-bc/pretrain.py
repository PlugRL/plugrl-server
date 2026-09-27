"""Behaviour-clone fpo-policy on DPPO's robomimic square demonstrations.

    python pretrain.py --data square_train.npz --norm normalization.npz --out DIR
                       [--steps 400000] [--batch 256] [--device cuda] [--seed 0]

The start FPO++ and DPPO both fine-tune from on square: a flow policy trained
on demonstrations. There is none for fpo-policy, so this makes one, from the
same 300 demonstrations DPPO pretrained its own policy on (robomimic's
multi-human square set, as DPPO processed it: `train.npz`, states and
actions scaled to [-1, 1] by `normalization.npz`).

* States and actions are taken back to the environment's units with
  `normalization.npz`, because fpo-policy normalises observations itself
  (running statistics, started here from the data's) and acts through a tanh.
* The target is the chunk of the next 4 actions - DPPO's horizon - through
  atanh, clipped at 0.999: the latent fpo-policy's tanh maps onto the action.
  Chunks run inside one demonstration; none is padded.
* The loss is the conditional flow-matching loss FPO++ scores with: the
  velocity error, one uniform t per sample (`compute_cfm_loss`, output mode
  u).
* AdamW, cosine decay, and an exponential moving average of the weights,
  which is what is saved - as DPPO and FPO++ both start from theirs.

Writes DIR/<step>/model.safetensors, the whole policy's state (the critic
at its initialisation), which `--algo.restore model` or `except-critic`
starts a run from, and DIR/pretrain.json with the settings and the loss.
"""

import argparse
import copy
import json
import math
import pathlib
import time

import numpy as np
import safetensors.torch
import torch

from plugrl_server.algorithm.fpo.utils import compute_cfm_loss
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
OBS_DIM, ACTION_DIM, HORIZON = 23, 7, 4
HIDDEN = (1024, 1024, 1024)  # the width and depth of DPPO's square network
CLIP = 0.999


def unnormalise(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def chunk_starts(lengths: np.ndarray, horizon: int) -> np.ndarray:
    """Every step whose next `horizon` actions lie in the same demonstration."""
    starts, offset = [], 0
    for length in lengths.astype(int):
        starts.extend(range(offset, offset + length - horizon + 1))
        offset += length
    return np.asarray(starts)


def policy_config(device: str) -> FPOPolicyConfig:
    return FPOPolicyConfig(
        device=device,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        hidden_dims=HIDDEN,
        state_keys=KEYS,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--norm", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data, norm = np.load(args.data), np.load(args.norm)
    states = unnormalise(data["states"], norm["obs_min"], norm["obs_max"])
    actions = unnormalise(data["actions"], norm["action_min"], norm["action_max"])
    starts = chunk_starts(data["traj_lengths"], HORIZON)
    chunks = np.stack([actions[starts + k] for k in range(HORIZON)], axis=1)
    latent = np.arctanh(np.clip(chunks, -CLIP, CLIP))
    device = torch.device(args.device)
    obs_t = torch.as_tensor(states[starts], dtype=torch.float32, device=device)
    latent_t = torch.as_tensor(latent, dtype=torch.float32, device=device)
    n = len(starts)
    print(
        f"{len(data['traj_lengths'])} demonstrations, {len(states)} steps, {n} chunks; "
        f"actions in [{actions.min():.3f}, {actions.max():.3f}], "
        f"{(np.abs(chunks) >= CLIP).mean():.4f} of elements clipped to +-{CLIP}"
    )

    policy = FPOPolicy(policy_config(args.device))
    # The data's statistics, as the running normaliser would have them.
    policy.update_obs_stats(torch.as_tensor(states, dtype=torch.float32, device=device))
    ema = copy.deepcopy(policy.actor).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.actor.parameters(), lr=args.lr, weight_decay=1e-6
    )

    def rate(step: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * step / args.steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    losses, window, started = [], [], time.time()
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, args.batch), device=device)
        obs, target = obs_t[idx], latent_t[idx]
        loss = compute_cfm_loss(
            policy,
            obs,
            target,
            output_mode="u",
            loss_eps=torch.randn(args.batch, 1, HORIZON, ACTION_DIM, device=device),
            loss_t=torch.rand(args.batch, 1, 1, device=device),
            obs_cache=policy.build_obs_cache(obs),
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), policy.actor.parameters()):
                e.lerp_(p, 1.0 - args.ema)
        window.append(float(loss))
        if step % args.log_every == 0:
            losses.append([step, sum(window) / len(window)])
            window = []
            print(
                f"step {step:7d}  loss {losses[-1][1]:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - started:.0f}s",
                flush=True,
            )

    policy.actor.load_state_dict(ema.state_dict())
    out = args.out / str(args.steps)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    safetensors.torch.save_file(state, out / "model.safetensors")
    (args.out / "pretrain.json").write_text(
        json.dumps(
            dict(
                data=str(args.data),
                norm=str(args.norm),
                chunks=n,
                horizon=HORIZON,
                hidden=HIDDEN,
                steps=args.steps,
                batch=args.batch,
                lr=args.lr,
                ema=args.ema,
                seed=args.seed,
                device=args.device,
                seconds=round(time.time() - started),
                loss=losses,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out / 'model.safetensors'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
