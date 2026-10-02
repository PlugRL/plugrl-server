"""E53: SB3's PPO with a CNN on 16 Atari environments, in its own process or behind the bridge.

    python train.py --arm inprocess --seed 0 --out runs/...
    python train.py --arm bridge    --seed 0 --out runs/... [--port 8830]

E50's design with image observations. The arms differ only in where the 16
`AtariWrapper(gym.make("ALE/Pong-v5"))` environments are stepped:
- inprocess: in SB3's DummyVecEnv, in this process;
- bridge: in atari_client.py, another process, on one connection to
  plugrl-bridges' `PlugRLVecEnv(image_key="frame")`, which sends each
  84x84x1 uint8 frame as it was rendered.

Everything else is one code path: VecMonitor, VecFrameStack(4), then PPO with
CnnPolicy at rl-baselines3-zoo's Atari settings (n_steps 128, 4 epochs,
minibatch 256, lr 2.5e-4, clip 0.1, entropy 0.01, value 0.5), held constant
rather than annealed. 40,960 steps (20 rollouts of 16 x 128), four torch
threads, CPU. This is far too short to learn Pong; E53 compares weights, not
returns. result.json gets the SHA-256 of the final policy and the time
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

import ale_py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

gym.register_envs(ale_py)
HERE = pathlib.Path(__file__).resolve().parent
ENV_ID = "ALE/Pong-v5"
NUM_ENVS = 16
TOTAL = 40_960
THREADS = 4
SETTINGS = dict(
    n_steps=128,
    n_epochs=4,
    batch_size=256,
    learning_rate=2.5e-4,
    clip_range=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
)


def weights_sha256(model: PPO) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["inprocess", "bridge"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8830)
    a = p.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)

    client = None
    if a.arm == "inprocess":
        env = DummyVecEnv([lambda: AtariWrapper(gym.make(ENV_ID))] * NUM_ENVS)
    else:
        from plugrl_bridges.sb3 import PlugRLVecEnv

        client_log = open(out / "client.log", "w", encoding="utf-8")
        client = subprocess.Popen(
            [sys.executable, str(HERE / "atari_client.py"), "--port", str(a.port),
             "--num-envs", str(NUM_ENVS), "--seed", str(a.seed), "--env-id", ENV_ID],
            stdout=client_log, stderr=subprocess.STDOUT,
        )  # fmt: skip
        actions = gym.make(ENV_ID).action_space
        env = PlugRLVecEnv(
            num_envs=NUM_ENVS,
            action_space=actions,
            port=a.port,
            host="127.0.0.1",
            image_key="frame",
            episode_log=str(out / "episodes.csv"),
        )
    env = VecMonitor(env, filename=str(out / "monitor.csv"))
    env = VecFrameStack(env, n_stack=4)
    # PPO(seed=) seeds torch and numpy, and calls env.seed(seed), which makes
    # DummyVecEnv reset env i with seed + i. atari_client.py does the same.
    model = PPO("CnnPolicy", env, seed=a.seed, device="cpu", verbose=0, **SETTINGS)
    start = time.perf_counter()
    model.learn(total_timesteps=TOTAL)
    learn_s = time.perf_counter() - start
    result = {
        "env": ENV_ID,
        "arm": a.arm,
        "seed": a.seed,
        "weights_sha256": weights_sha256(model),
        "learn_s": round(learn_s, 3),
        "steps": model.num_timesteps,
        "steps_per_s": round(model.num_timesteps / learn_s, 1),
        "observation_space": str(env.observation_space),
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
