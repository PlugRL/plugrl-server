"""E53's env client: N Atari environments behind SB3's AtariWrapper, on one connection.

    python atari_client.py --port 8830 --num-envs 16 --seed 0 [--env-id ALE/Pong-v5]

E50's gym_client.py, with three changes:
- Each env is `AtariWrapper(gym.make(env_id))`, as SB3 preprocesses Atari:
  no-op starts, frame skip 4, 84x84 grayscale, a life lost ends the episode,
  and rewards clipped to their sign.
- The observation it sends is that frame, a uint8 [n, 84, 84, 1] image
  named `frame`, with no states.
- The action arrives as one float per env (the protocol's actions are
  float), and is applied as the integer it is.

Env i is reset with seed + i first and with no seed afterwards, as SB3's
DummyVecEnv does after `VecEnv.seed(seed)`. An env whose episode ended is
reset before the next infer, as DummyVecEnv resets it inside the step.
"""

from __future__ import annotations

import argparse
import time

import ale_py
import gymnasium as gym
import numpy as np
import websockets.exceptions
import websockets.sync.client
from stable_baselines3.common.atari_wrappers import AtariWrapper

from plugrl_protocol import msgpack_numpy

gym.register_envs(ale_py)
STOP_REASON = "plugrl-server-stop"


def observation(frames: np.ndarray) -> dict:
    return {"images": {"frame": frames}, "states": {}, "text": "atari"}


def connect(uri: str, deadline: float):
    while True:
        try:
            return websockets.sync.client.connect(
                uri, compression=None, max_size=None, open_timeout=10
            )
        except (OSError, websockets.exceptions.InvalidHandshake):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8830)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="ALE/Pong-v5")
    a = p.parse_args()

    n = a.num_envs
    envs = [AtariWrapper(gym.make(a.env_id)) for _ in range(n)]
    obs = np.stack([env.reset(seed=a.seed + i)[0] for i, env in enumerate(envs)])
    indices = np.arange(n, dtype=np.int64)
    step_ids = np.zeros(n, dtype=np.int64)
    packer = msgpack_numpy.Packer()
    steps = 0

    ws = connect(f"ws://{a.host}:{a.port}", time.monotonic() + 120)
    with ws:
        msgpack_numpy.unpackb(ws.recv())  # the metadata; nothing in it is needed
        try:
            while True:
                ws.send(
                    packer.pack(
                        {
                            "message_type": "infer",
                            "data": observation(obs),
                            "env_indices": indices,
                            "step_ids": step_ids.copy(),
                        }
                    )
                )
                reply = msgpack_numpy.unpackb(ws.recv())
                actions = np.asarray(reply["data"]["action"])[0]  # [H, n, 1] -> [n, 1]
                ended = np.empty_like(obs)
                after = np.empty_like(obs)
                rewards = np.zeros(n, dtype=np.float32)
                terminated = np.zeros(n, dtype=np.bool_)
                truncated = np.zeros(n, dtype=np.bool_)
                sent_ids = step_ids.copy()
                for i, env in enumerate(envs):
                    o, r, te, tr, _ = env.step(int(round(float(actions[i][0]))))
                    ended[i] = o
                    rewards[i], terminated[i], truncated[i] = r, te, tr
                    if te or tr:
                        after[i] = env.reset()[0]
                        step_ids[i] = 0
                    else:
                        after[i] = o
                        step_ids[i] += 1
                ws.send(
                    packer.pack(
                        {
                            "message_type": "feedback",
                            "env_indices": indices,
                            "step_ids": sent_ids,
                            "data": {
                                "obs": observation(ended),
                                "rewards": rewards,
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": {},
                            },
                        }
                    )
                )
                obs = after
                steps += 1
        except websockets.exceptions.ConnectionClosed as closed:
            reason = closed.rcvd.reason if closed.rcvd is not None else ""
            print(f"closed after {steps} steps: {reason!r}", flush=True)
            return 0 if reason == STOP_REASON else 1


if __name__ == "__main__":
    raise SystemExit(main())
