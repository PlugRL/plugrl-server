"""E50's env client: N gymnasium environments on one connection, and nothing else.

    python gym_client.py --port 8820 --num-envs 16 --seed 0 [--env-id Pendulum-v1]

Env i is reset with seed + i on its first reset and with no seed afterwards,
as SB3's DummyVecEnv does after `VecEnv.seed(seed)`. One connection carries
all N, so the bridge gives env i slot i. Each step: an infer with every env's
observation, the action chunk's first step applied to each env, and a
feedback with the observation the step ended on (the terminal one, if it
ended), the reward as float32, and the flags. An env whose episode ended is
reset before the next infer. No reuse-feedback-obs, no info, no reconnect:
the bridge has no reason to resync a client that keeps lockstep, so a close
that is not plugrl-server-stop exits 1.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import websockets.exceptions
import websockets.sync.client

from plugrl_protocol import msgpack_numpy

STOP_REASON = "plugrl-server-stop"


def observation(obs: np.ndarray) -> dict:
    return {"images": {}, "states": {"obs": obs}, "text": "gym"}


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
    p.add_argument("--port", type=int, default=8820)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="Pendulum-v1")
    a = p.parse_args()

    n = a.num_envs
    envs = [gym.make(a.env_id) for _ in range(n)]
    obs = np.stack([env.reset(seed=a.seed + i)[0] for i, env in enumerate(envs)])
    obs = obs.astype(np.float32)
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
                actions = np.asarray(reply["data"]["action"])[0]  # [H, n, d] -> [n, d]
                ended = np.empty_like(obs)
                after = np.empty_like(obs)
                rewards = np.zeros(n, dtype=np.float32)
                terminated = np.zeros(n, dtype=np.bool_)
                truncated = np.zeros(n, dtype=np.bool_)
                sent_ids = step_ids.copy()
                for i, env in enumerate(envs):
                    o, r, te, tr, _ = env.step(actions[i])
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
