"""E47's round-trip client: one env, HalfCheetah-sized observations, no simulator.

E46's explore_variant_client.py with only its v1 variant: every infer
carries the observation, and every action is answered with a feedback.

    python rtt_client.py PORT EXCHANGES

Prints the mean infer round trip in ms.
"""

import sys
import time

import numpy as np
import websockets.sync.client

from plugrl_protocol import msgpack_numpy

port, exchanges = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
packer = msgpack_numpy.Packer()


def obs() -> dict:
    return {
        "images": {},
        "states": {"obs": rng.normal(size=(1, 17))},
        "text": np.asarray([""]),
    }


current = obs()
total = 0.0
with websockets.sync.client.connect(
    f"ws://127.0.0.1:{port}", compression=None, max_size=None
) as ws:
    ws.recv()
    for step in range(exchanges):
        message = {
            "message_type": "infer",
            "data": current,
            "env_indices": np.asarray([0]),
            "step_ids": np.asarray([step]),
        }
        t0 = time.perf_counter()
        ws.send(packer.pack(message))
        ws.recv()
        total += time.perf_counter() - t0
        current = obs()
        ws.send(
            packer.pack(
                {
                    "message_type": "feedback",
                    "env_indices": np.asarray([0]),
                    "step_ids": np.asarray([step]),
                    "data": {
                        "obs": current,
                        "rewards": np.ones(1, np.float32),
                        "terminated": np.zeros(1, bool),
                        "truncated": np.zeros(1, bool),
                        "info": {},
                    },
                }
            )
        )
print(f"{total / exchanges * 1e3:.4f}")
