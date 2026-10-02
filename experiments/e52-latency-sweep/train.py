"""E52: E50's train.py, with the client able to connect through a delay proxy.

    python train.py --env pendulum --arm bridge --seed 0 --out runs/... \
        --port 8820 --client-port 8821

The one change from E50's file: --client-port, the port the env client
connects to (default: --port, as in E50). E52 points it at delay_proxy.py,
which forwards to the bridge on --port. The client is E50's gym_client.py.
What follows is E50's description, unchanged.

    python train.py --env pendulum --arm inprocess --seed 0 --out runs/...
    python train.py --env pendulum --arm bridge    --seed 0 --out runs/... [--port 8820]

The two arms differ only in where the 16 environments are stepped:
- inprocess: SB3's DummyVecEnv of gym.make(ENV_ID), in this process;
- bridge: plugrl-bridges' PlugRLVecEnv, with the same 16 environments in
  gym_client.py, another process, on one connection.

Everything else is one code path: VecMonitor, then PPO, 16 x 256 frames per
rollout, one torch thread, CPU.
- pendulum: Pendulum-v1, 102,400 steps, rl-baselines3-zoo's settings as
  plugrl-bridges' examples/sb3/ppo.py --preset pendulum uses them.
- halfcheetah: HalfCheetah-v5, 409,600 steps, SB3's defaults with no
  VecNormalize. Its observations are float64; gym_client.py sends them as
  float32.
result.json gets the SHA-256 of the final policy's parameters and the time
model.learn() took.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

HERE = pathlib.Path(__file__).resolve().parent
CLIENT = HERE.parent / "e50-boundary-transparency" / "gym_client.py"
NUM_ENVS = 16
ENVS = {
    "pendulum": dict(
        id="Pendulum-v1",
        total=102_400,
        action=(1, -2.0, 2.0),
        # plugrl-bridges examples/sb3/ppo.py, preset "pendulum".
        ppo=dict(
            n_steps=256,
            gamma=0.9,
            gae_lambda=0.95,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
        ),
    ),
    "halfcheetah": dict(
        id="HalfCheetah-v5",
        total=409_600,
        action=(6, -1.0, 1.0),
        ppo=dict(n_steps=256),
    ),
}


def weights_sha256(model: PPO) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(ENVS), required=True)
    p.add_argument("--arm", choices=["inprocess", "bridge"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8820)
    p.add_argument("--client-port", type=int, default=None)
    a = p.parse_args()
    client_port = a.client_port or a.port
    spec = ENVS[a.env]
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    client = None
    if a.arm == "inprocess":
        env = DummyVecEnv([lambda: gym.make(spec["id"])] * NUM_ENVS)
    else:
        from plugrl_bridges.sb3 import PlugRLVecEnv

        dim, low, high = spec["action"]
        client_log = open(out / "client.log", "w", encoding="utf-8")
        client = subprocess.Popen(
            [sys.executable, str(CLIENT), "--port", str(client_port),
             "--num-envs", str(NUM_ENVS), "--seed", str(a.seed),
             "--env-id", spec["id"]],
            stdout=client_log, stderr=subprocess.STDOUT,
        )  # fmt: skip
        env = PlugRLVecEnv(
            num_envs=NUM_ENVS,
            action_space=gym.spaces.Box(low, high, (dim,), np.float32),
            port=a.port,
            host="127.0.0.1",
            episode_log=str(out / "episodes.csv"),
        )
    env = VecMonitor(env, filename=str(out / "monitor.csv"))
    # PPO(seed=) seeds torch and numpy, and calls env.seed(seed), which makes
    # DummyVecEnv reset env i with seed + i. gym_client.py does the same.
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, **spec["ppo"])
    start = time.perf_counter()
    model.learn(total_timesteps=spec["total"])
    learn_s = time.perf_counter() - start
    result = {
        "env": a.env,
        "arm": a.arm,
        "seed": a.seed,
        "weights_sha256": weights_sha256(model),
        "learn_s": round(learn_s, 3),
        "steps": model.num_timesteps,
        "steps_per_s": round(model.num_timesteps / learn_s, 1),
        "torch": torch.__version__,
    }
    env.close()
    if client is not None:
        result["client_rc"] = client.wait(timeout=60)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
