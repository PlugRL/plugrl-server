"""E57: evaluate one run's final policy on a clean environment.

    python evaluate.py RUN_DIR OUT_DIR [--episodes N]

RUN_DIR is an E54 or E56 run directory (model.zip, result.json). The policy
acts deterministically in a fresh gymnasium environment of the same id, at
its default settings, for N episodes reset with seeds 10000, 10001, ... No
bridge, no env client, no fault. Writes OUT_DIR/clean_eval.json.
"""

import argparse
import json
import pathlib
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

IDS = {
    "pendulum": "Pendulum-v1",
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "pusher": "Pusher-v5",
}
SEED0 = 10_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(1)

    res = json.loads((a.run / "result.json").read_text())
    env = gym.make(IDS[res["env"]])
    model = PPO.load(a.run / "model.zip", device="cpu")

    returns, lengths = [], []
    t0 = time.time()
    for k in range(a.episodes):
        obs, _ = env.reset(seed=SEED0 + k)
        done, ret, n = False, 0.0, 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(act)
            ret += float(r)
            n += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(n)

    out = dict(
        env=res["env"],
        fault=res["fault"],
        dose=res["dose"],
        seed=res["seed"],
        weights_sha256=res["weights_sha256"],
        episodes=a.episodes,
        returns=returns,
        lengths=lengths,
        mean=float(np.mean(returns)),
        eval_s=round(time.time() - t0, 3),
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "clean_eval.json").write_text(json.dumps(out))
    print(
        json.dumps(
            {k: out[k] for k in ("env", "fault", "dose", "seed", "mean", "eval_s")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
