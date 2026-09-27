"""Record evaluation episodes of one coverage cell's final checkpoint.

    python record.py CELL [--episodes 5] [--port 9700] [--out media/raw]

An eval server loaded with the cell's final checkpoint (`cells.json`), and an
env client with the recorder on, for EPISODES episodes. The server runs this
checkout's code through PYTHONPATH; the Pythons and run directories are found
under PLUGRL_HOME (default ~/zuogou/plugrl, the layout on `guangzhao`).

The eval algorithm acts without the sampling noise training used: the flow
policy integrates its ODE, DPPO's policy samples its own denoising chain.
MuJoCo clients are seeded at 100, outside the training seeds; robomimic's
refuses a seed.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HOME = pathlib.Path(os.environ.get("PLUGRL_HOME", "~/zuogou/plugrl")).expanduser()
SERVER_PY = HOME / "plugrl-server/.venv/bin/python"
CLIENT_DIR = HOME / "plugrl-env-client"
CLIENT_PY = {
    "mujoco-v1": CLIENT_DIR / ".venv/bin/python",
    "robomimic-v1": CLIENT_DIR / ".venv-robomimic/bin/python",
}
EVAL_SEED = 100


def final_checkpoint(cell: dict) -> pathlib.Path:
    run = HOME / cell["runs"].format(seed=cell["seed"])
    steps = sorted(
        (p for p in run.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
    )
    if not steps:
        raise SystemExit(f"{cell['id']}: no checkpoint under {run}")
    return steps[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "media/raw")
    args = parser.parse_args()

    cells = {c["id"]: c for c in json.loads((HERE / "cells.json").read_text())["cells"]}
    cell = cells[args.cell]
    ckpt = final_checkpoint(cell)
    out = (args.out / cell["id"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT / "src"))

    server_cmd = [
        str(SERVER_PY), "-m", "plugrl_server.cli", *cell["server"],
        "--algo.policy-checkpoint-path", str(ckpt),
        "--algo.num-episodes", str(args.episodes),
        "--port", str(args.port), "--seed", "0", "--policy.device", "cpu",
        "--no-show-progress-bar", "--no-show-metric-table",
        "--checkpoint-base-dir", str(out / "server"), "--exp-name", cell["id"], "--overwrite",
    ]  # fmt: skip
    family = cell["client"][0]
    client_cmd = [
        str(CLIENT_PY[family]), "-m", "plugrl_env_client.cli", *cell["client"],
        "--server-host", "127.0.0.1", "--server-port", str(args.port),
        "--num-envs", "1", "--num-episodes", str(args.episodes),
        "--runner.replan-steps", str(cell["replan"]),
        "--recorder.episode-freq", "1", "--recorder.record-video",
        "--exp-name", cell["id"],
    ]  # fmt: skip
    if family == "mujoco-v1":
        client_cmd += ["--runner.seed", str(EVAL_SEED)]
    client_env = dict(env, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")

    (out / "record.json").write_text(
        json.dumps(
            {"cell": cell["id"], "checkpoint": str(ckpt), "episodes": args.episodes,
             "server": server_cmd[1:], "client": client_cmd[1:]},
            indent=2,
        )
    )  # fmt: skip
    print(f"{cell['id']}: {ckpt}")
    with open(out / "server.log", "w") as slog:
        server = subprocess.Popen(
            server_cmd, cwd=ROOT, env=env, stdout=slog, stderr=subprocess.STDOUT
        )
    try:
        for _ in range(300):
            if "is listening" in (out / "server.log").read_text(errors="replace"):
                break
            if server.poll() is not None:
                raise SystemExit(
                    f"{cell['id']}: the server exited; see {out / 'server.log'}"
                )
            time.sleep(1)
        with open(out / "client.log", "w") as clog:
            rc = subprocess.run(
                client_cmd, cwd=out, env=client_env, stdout=clog,
                stderr=subprocess.STDOUT, timeout=1800,
            ).returncode  # fmt: skip
        print(f"{cell['id']}: client exited {rc}")
    finally:
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=30)
    return rc


if __name__ == "__main__":
    sys.exit(main())
