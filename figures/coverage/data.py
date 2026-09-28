"""Gather the figure's data into one file for the project page.

    python data.py [--out media/site/coverage.json]

For every cell in cells.json: its status, experiments and note; the return
per iteration of all three seeds, or the column's `metric` if it names one,
from their tensorboards (preceded by the
runs they resumed from, if `resumed_from` names them); and what pick.py
recorded about the clip, if it has run; and from commands.json, the two
commands that trained it and the env-client environment it needs. The `vla`
section of cells.json is copied as it stands, with the same additions. Needs
tensorboard - the server's venv has it.
"""

import argparse
import json
import os
import pathlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path(os.environ.get("PLUGRL_HOME", "~/zuogou/plugrl")).expanduser()


def returns(run: pathlib.Path, tag: str = "rollout/reward") -> list[float]:
    events = sorted((run / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    digits = 1 if tag == "rollout/reward" else 3
    return [round(e.value, digits) for e in acc.Scalars(tag)]


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

    # A task whose runs do not share a reward plots a measure they do share.
    metric = {c["id"]: c.get("metric", "rollout/reward") for c in spec["columns"]}
    cells = []
    for cell in spec["cells"]:
        # Most cells ran seeds 0-2; a cell that ran others names them.
        seeds = cell.get("seeds", [0, 1, 2])
        tag = metric[cell["column"]]
        curves = [returns(HOME / cell["runs"].format(seed=s), tag) for s in seeds]
        # A cell whose runs resumed from earlier ones plots the whole run.
        if "resumed_from" in cell:
            curves = [
                returns(HOME / cell["resumed_from"].format(seed=s), tag) + curve
                for s, curve in zip(seeds, curves)
            ]
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

    # The two commands behind each cell, and the env-client environment each
    # needs. Every cell must have them, and every script they cite must exist.
    commands = json.loads((HERE / "commands.json").read_text())
    for cell in cells + (vla or {}).get("cells", []):
        train = commands["cells"].get(cell["id"])
        if train is None:
            raise SystemExit(f"{cell['id']}: no commands in commands.json")
        if not (HERE.parent.parent / train["source"]).is_file():
            raise SystemExit(f"{cell['id']}: {train['source']} does not exist")
        if train["env"] not in commands["envs"]:
            raise SystemExit(f"{cell['id']}: unknown env {train['env']}")
        cell["train"] = train
    data = {
        "rows": spec["rows"],
        "columns": spec["columns"],
        "ranges": ranges,
        "cells": cells,
        "vla": vla,
        "experiments": {e: dirs[e] for e in sorted(wanted, key=lambda e: int(e[1:]))},
        "envs": commands["envs"],
        "server_packages": commands["server_packages"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(f"wrote {args.out}: {len(cells)} cells, {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
