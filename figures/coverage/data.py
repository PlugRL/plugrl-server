"""Gather the figure's data into one file for the project page.

    python data.py [--out media/site/coverage.json]

For every cell in cells.json: its status, experiments and note; the return
per iteration of all three seeds, from their tensorboards; and what pick.py
recorded about the clip, if it has run. The `vla` section of cells.json is
copied as it stands, with its clips' notes merged the same way. Needs
tensorboard - the server's venv has it.
"""

import argparse
import json
import os
import pathlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path(os.environ.get("PLUGRL_HOME", "~/zuogou/plugrl")).expanduser()


def returns(run: pathlib.Path) -> list[float]:
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    if "rollout/reward" not in acc.Tags()["scalars"]:
        return []
    return [round(e.value, 1) for e in acc.Scalars("rollout/reward")]


def clip(site: pathlib.Path, cell_id: str) -> dict | None:
    meta = site / f"{cell_id}.json"
    return json.loads(meta.read_text()) if meta.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=pathlib.Path, default=HERE / "media/site/coverage.json"
    )
    args = parser.parse_args()
    spec = json.loads((HERE / "cells.json").read_text())
    site = args.out.parent

    cells = []
    for cell in spec["cells"]:
        curves = [returns(HOME / cell["runs"].format(seed=s)) for s in range(3)]
        cells.append(
            {
                "id": cell["id"],
                "row": cell["row"],
                "column": cell["column"],
                "status": cell["status"],
                "experiments": cell["experiments"],
                "note": cell["note"],
                "note_zh": cell["note_zh"],
                "seed": cell["seed"],
                "curves": curves,
                "clip": clip(site, cell["id"]),
            }
        )
    # One vertical range per task, shared by its rows, so a curve that does
    # not move looks flat beside one that does.
    ranges = {}
    for col in spec["columns"]:
        values = [
            v for c in cells if c["column"] == col["id"] for s in c["curves"] for v in s
        ]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
        ranges[col["id"]] = [lo, hi if hi > lo else lo + 1.0]
    vla = spec.get("vla")
    if vla:
        vla = {k: v for k, v in vla.items() if not k.startswith("_")}
        for cell in vla["cells"]:
            cell["clip"] = clip(site, cell["id"])
            for key in ("video_key", "native_fps", "whole"):
                cell.pop(key, None)
    # Experiment directories on `main`, for the links: E6 -> e6-first-learning-curve.
    dirs = {}
    for d in sorted((HERE.parent.parent / "experiments").iterdir()):
        head = d.name.split("-")[0]
        if d.is_dir() and head[:1] == "e" and head[1:].isdigit():
            dirs.setdefault(f"E{int(head[1:])}", d.name)
    wanted = {e for c in cells for e in c["experiments"]} | {
        e for c in (vla or {}).get("cells", []) for e in c["experiments"]
    }
    missing = sorted(wanted - dirs.keys())
    if missing:
        raise SystemExit(f"no experiment directory for {missing}")
    data = {
        "rows": spec["rows"],
        "columns": spec["columns"],
        "ranges": ranges,
        "cells": cells,
        "vla": vla,
        "experiments": {e: dirs[e] for e in sorted(wanted, key=lambda e: int(e[1:]))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(f"wrote {args.out}: {len(cells)} cells, {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
