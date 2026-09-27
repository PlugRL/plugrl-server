"""Choose a recorded cell's median episode and encode its clip and still.

    python pick.py CELL [--raw media/raw] [--out media/site]

Of the episodes `record.py` recorded, the one whose return is the median
(ties: the earlier episode) - not the best. Its frames are resampled to real
time at no more than 25 fps, cut to the first 8 seconds, scaled to 256
pixels and encoded as H.264; an episode shorter than 3 seconds is slowed to
fill 3, and the factor is recorded. Writes CELL.mp4, CELL.jpg (the frame 40%
of the way in) and CELL.json (which episode, of how many, every return).
Needs imageio with its ffmpeg, and Pillow - the env client's venv has both.
"""

import argparse
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
SIZE = 256
MAX_FPS = 25.0
MAX_SECONDS = 8.0
MIN_SECONDS = 3.0


def episodes(raw: pathlib.Path, cell: str) -> list[dict]:
    root = raw / cell / "runs" / cell / "rollout" / "proc_000" / "sampled"
    found = []
    for ep in sorted(root.glob("ep_*")):
        env0 = ep / "env_000"
        info = json.loads((env0 / "obs" / "last" / "info.json").read_text())["episode"]
        videos = sorted((env0 / "images").glob("*.mp4"))
        if videos:
            found.append(
                {"index": int(ep.name.split("_")[1]), "return": float(info["r"]),
                 "length": int(info["l"]), "success": bool(info["s"]), "video": videos[0]}
            )  # fmt: skip
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell")
    parser.add_argument("--raw", type=pathlib.Path, default=HERE / "media/raw")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "media/site")
    args = parser.parse_args()
    cell = {c["id"]: c for c in json.loads((HERE / "cells.json").read_text())["cells"]}[
        args.cell
    ]
    eps = episodes(args.raw, args.cell)
    if not eps:
        raise SystemExit(f"{args.cell}: no recorded episodes under {args.raw}")
    ranked = sorted(eps, key=lambda e: (e["return"], e["index"]))
    chosen = ranked[(len(ranked) - 1) // 2]

    native = float(cell["native_fps"])
    step = max(1, round(native / MAX_FPS))
    real_fps = native / step
    frames = [f for f in imageio.get_reader(chosen["video"])][::step]
    frames = frames[: int(MAX_SECONDS * real_fps)]
    seconds = len(frames) / real_fps
    play_fps = (
        real_fps if seconds >= MIN_SECONDS else max(len(frames) / MIN_SECONDS, 1.0)
    )
    frames = [
        np.asarray(Image.fromarray(f).resize((SIZE, SIZE), Image.LANCZOS))
        for f in frames
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.out / f"{args.cell}.mp4", fps=play_fps, codec="libx264", quality=None,
        pixelformat="yuv420p", macro_block_size=16,
        ffmpeg_params=["-crf", "30", "-preset", "slow", "-movflags", "+faststart"],
    )  # fmt: skip
    for f in frames:
        writer.append_data(f)
    writer.close()
    Image.fromarray(frames[int(0.4 * (len(frames) - 1))]).save(
        args.out / f"{args.cell}.jpg", quality=82
    )
    meta = {
        "cell": args.cell,
        "episode": chosen["index"],
        "of": len(eps),
        "return": round(chosen["return"], 1),
        "length": chosen["length"],
        "success": chosen["success"],
        "returns": [round(e["return"], 1) for e in eps],
        "seconds": round(len(frames) / play_fps, 2),
        "speed": round(play_fps / real_fps, 3),
    }
    (args.out / f"{args.cell}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
