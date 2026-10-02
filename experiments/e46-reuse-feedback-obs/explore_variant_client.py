"""E46 exploration: which part of a v2 infer makes plugrl-server answer sooner?

Drives one plugrl-server (fpo-policy) with HalfCheetah-sized observations,
one env, no simulator, and prints the mean infer round trip:

    python explore_variant_client.py PORT VARIANT EXCHANGES

    v1        the observation in every infer
    v2        reuse=[True] and an empty batch after the first exchange
    v2-full   reuse=[False] with the observation sent in full
"""

import sys
import time

import numpy as np
import websockets.sync.client

from plugrl_protocol import msgpack_numpy

port, variant, exchanges = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(0)
packer = msgpack_numpy.Packer()


def obs(rows: int) -> dict:
    return {
        "images": {},
        "states": {"obs": rng.normal(size=(rows, 17))},
        "text": np.asarray([""] * rows),
    }


current = obs(1)
total = 0.0
with websockets.sync.client.connect(
    f"ws://127.0.0.1:{port}", compression=None, max_size=None
) as ws:
    ws.recv()
    for step in range(exchanges):
        infer = {
            "message_type": "infer",
            "env_indices": np.asarray([0]),
            "step_ids": np.asarray([step]),
        }
        if variant == "v2" and step > 0:
            infer["data"] = {
                "images": {},
                "states": {"obs": np.zeros((0, 17))},
                "text": np.asarray([], dtype="<U1"),
            }
            infer["reuse"] = np.asarray([True])
        else:
            infer["data"] = current
            if variant == "v2-full":
                infer["reuse"] = np.asarray([False])
        t0 = time.perf_counter()
        ws.send(packer.pack(infer))
        ws.recv()
        total += time.perf_counter() - t0
        current = obs(1)
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
print(f"{variant}\t{total / exchanges * 1e3:.3f} ms per infer round trip")
