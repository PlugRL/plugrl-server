"""E46's boundary-cost client: E43's bench_client.py, with `--reuse`.

E43's client, unchanged in what it sends without the flag: per exchange, an
infer carrying the observation, the action back, and a feedback carrying the
observation again. The same observation is sent each time, and no episode
ends.

With `--reuse` it uses SPEC.md section 10.1's `reuse-feedback-obs`. After
the first exchange, each infer marks its one row in `reuse` and sends an
empty batch, because the server holds that observation from the feedback
just sent. It refuses to run against a server that does not list the
feature, so a run cannot silently measure the wrong thing.

    python bench_client.py --host H --port P --steps 1000 --img 224 --cameras 2 [--reuse]

--img 0 sends states only (17 floats). Otherwise a base camera of IMG x IMG,
and with --cameras 2 a wrist camera of half that: 128/1 is 48 KiB, 224/2
184 KiB, 448/1 588 KiB. Prints one tab-separated line:

    exchanges  pack_ms  rtt_ms  unpack_ms  wall_s  exchanges_per_s  infer_bytes  feedback_bytes

where each *_ms is the mean over the exchanges, and the two byte counts are
the size of the last infer and feedback frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from array import array

import msgpack
import websockets.sync.client

INFER, FEEDBACK, METADATA, ACTION = "infer", "feedback", "metadata", "action"
REUSE_FEEDBACK_OBS = "reuse-feedback-obs"
_ARRAY_CODE = {("f", 4): "f", ("f", 8): "d", ("i", 8): "q", ("u", 1): "B"}


def encode_array(values, typestr: str, shape: tuple[int, ...]) -> dict:
    order, kind, size = typestr[0], typestr[1], int(typestr[2:])
    if kind == "b":
        raw = bytes(1 if v else 0 for v in values)
    else:
        buf = array(_ARRAY_CODE[(kind, size)], values)
        if (sys.byteorder == "little") != (order in "<|"):
            buf.byteswap()
        raw = buf.tobytes()
    return {
        b"__ndarray__": True,
        b"data": raw,
        b"dtype": typestr,
        b"shape": list(shape),
    }


def decode_array(obj: dict) -> list:
    typestr = obj[b"dtype"]
    typestr = typestr.decode() if isinstance(typestr, bytes) else typestr
    order, kind, size = typestr[0], typestr[1], int(typestr[2:])
    buf = array(_ARRAY_CODE[(kind, size)])
    buf.frombytes(obj[b"data"])
    if (sys.byteorder == "little") != (order in "<|"):
        buf.byteswap()
    return list(buf)


def image(h: int, rows: int) -> dict:
    n = rows * h * h * 3
    raw = (bytes(range(256)) * (n // 256 + 1))[:n]
    return {
        b"__ndarray__": True,
        b"data": raw,
        b"dtype": "|u1",
        b"shape": [rows, h, h, 3],
    }


def observation(img: int, cameras: int, rows: int = 1) -> dict:
    """E43's observation, batched to `rows` (0 for an infer that reuses it)."""
    images = {}
    if img:
        images["base"] = image(img, rows)
        if cameras == 2:
            images["wrist"] = image(img // 2, rows)
    return {
        "images": images,
        "states": {"obs": encode_array([0.1] * 17 * rows, "<f8", (rows, 17))},
        "text": ["bench"] * rows,
    }


def run(host: str, port: int, steps: int, img: int, cameras: int, reuse: bool) -> int:
    packer = msgpack.Packer()
    env_indices = encode_array([0], "<i8", (1,))
    reused = encode_array([True], "|b1", (1,))
    pack_s = rtt_s = unpack_s = 0.0
    infer_bytes = feedback_bytes = 0
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=30
    ) as ws:
        meta = msgpack.unpackb(ws.recv(), raw=False, strict_map_key=False)
        if meta.get("message_type") != METADATA:
            print(
                f"expected {METADATA}, got {meta.get('message_type')!r}",
                file=sys.stderr,
            )
            return 1
        if reuse and REUSE_FEEDBACK_OBS not in (meta.get("data") or {}).get(
            "features", []
        ):
            print(f"the server does not offer {REUSE_FEEDBACK_OBS}", file=sys.stderr)
            return 1
        start = time.perf_counter()
        for step in range(steps):
            step_ids = encode_array([step], "<i8", (1,))
            t0 = time.perf_counter()
            infer = {
                "message_type": INFER,
                "env_indices": env_indices,
                "step_ids": step_ids,
            }
            if reuse and step > 0:
                # The server holds this env's observation from the feedback
                # just sent, and no episode ends here.
                infer["data"] = observation(img, cameras, rows=0)
                infer["reuse"] = reused
            else:
                infer["data"] = observation(img, cameras)
            msg = packer.pack(infer)
            t1 = time.perf_counter()
            ws.send(msg)
            reply = ws.recv()
            t2 = time.perf_counter()
            if isinstance(reply, str):
                print(f"server error: {reply}", file=sys.stderr)
                return 1
            action_msg = msgpack.unpackb(reply, raw=False, strict_map_key=False)
            if action_msg.get("message_type") != ACTION:
                print(
                    f"expected {ACTION}, got {action_msg.get('message_type')!r}",
                    file=sys.stderr,
                )
                return 1
            decode_array(action_msg["data"]["action"])
            t3 = time.perf_counter()
            pack_s += t1 - t0
            rtt_s += t2 - t1
            unpack_s += t3 - t2
            feedback = packer.pack(
                {
                    "message_type": FEEDBACK,
                    "env_indices": env_indices,
                    "step_ids": step_ids,
                    "data": {
                        "obs": observation(img, cameras),
                        "rewards": encode_array([0.0], "<f4", (1,)),
                        "terminated": encode_array([0], "|b1", (1,)),
                        "truncated": encode_array([0], "|b1", (1,)),
                        "info": {},
                    },
                }
            )
            ws.send(feedback)
            infer_bytes, feedback_bytes = len(msg), len(feedback)
        wall = time.perf_counter() - start
    ms = 1000.0 / steps
    print(
        f"{steps}\t{pack_s * ms:.4f}\t{rtt_s * ms:.4f}\t{unpack_s * ms:.4f}"
        f"\t{wall:.3f}\t{steps / wall:.1f}\t{infer_bytes}\t{feedback_bytes}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--img", type=int, default=0)
    p.add_argument("--cameras", type=int, default=1, choices=(1, 2))
    p.add_argument("--reuse", action="store_true", help="use reuse-feedback-obs")
    a = p.parse_args()
    return run(a.host, a.port, a.steps, a.img, a.cameras, a.reuse)


if __name__ == "__main__":
    raise SystemExit(main())
