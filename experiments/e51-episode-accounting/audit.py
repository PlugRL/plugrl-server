"""E51: does each episode counter count a known episode right? See PROTOCOL.md.

    python audit.py [--port 8830] [--env-client-src PATH]

Every counter runs on Counting-v0 (return 15, length 5), 2 envs, 3 episodes
each, and prints what it logged next to what PROTOCOL.md predicted.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from counting_env import LENGTH, Counting  # noqa: E402

N, EPISODES = 2, 3
RIGHT = [(15.0, 5)] * (N * EPISODES)
NEXT, SAME = AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP


def steps_for(mode) -> int:
    # Under NEXT_STEP each episode is followed by a reset step.
    return EPISODES * (LENGTH + (1 if mode == NEXT else 0))


def actions(n: int) -> np.ndarray:
    return np.zeros((n, 1), np.float32)


def masked(episode: dict, mask) -> list[tuple[float, int]]:
    return [
        (float(episode["r"][i]), int(episode["l"][i])) for i in np.flatnonzero(mask)
    ]


# ------------------------------------------------------------------ gymnasium


def vector_wrapper(mode):
    """A, B: gymnasium's vector RecordEpisodeStatistics."""
    env = gym.vector.SyncVectorEnv([Counting] * N, autoreset_mode=mode)
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    env.reset(seed=0)
    got = []
    for _ in range(steps_for(mode)):
        *_, infos = env.step(actions(N))
        if "episode" in infos:
            got += masked(infos["episode"], infos["_episode"])
    return got


def per_env_infos(mode) -> list[dict]:
    """C-F: CleanRL's layering, single-env RecordEpisodeStatistics per sub-env."""
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(Counting())] * N,
        autoreset_mode=mode,
    )
    env.reset(seed=0)
    return [env.step(actions(N))[-1] for _ in range(steps_for(mode))]


def per_env_wrapper(mode):
    got = []
    for infos in per_env_infos(mode):
        source = infos if mode == NEXT else infos.get("final_info", {})
        if "episode" in source:
            got += masked(source["episode"], source["_episode"])
    return got


def cleanrl_logging(mode):
    """E, F: the loop in CleanRL's ppo_continuous_action.py (master), verbatim
    but for print and writer.add_scalar, which append to a list instead."""
    logged = []
    for infos in per_env_infos(mode):
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    logged.append(
                        (float(info["episode"]["r"]), int(info["episode"]["l"]))
                    )
    return logged


# ------------------------------------------------------------------------ SB3


def sb3_vec_monitor():
    """G: SB3's VecMonitor over DummyVecEnv."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    env = VecMonitor(DummyVecEnv([Counting] * N))
    env.seed(0)
    env.reset()
    got = []
    for _ in range(EPISODES * LENGTH):
        *_, infos = env.step(actions(N))
        got += [
            (float(i["episode"]["r"]), int(i["episode"]["l"]))
            for i in infos
            if "episode" in i
        ]
    return got


def sb3_monitor():
    """H: SB3's Monitor per env inside DummyVecEnv."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: Monitor(Counting())] * N)
    env.seed(0)
    env.reset()
    got = []
    for _ in range(EPISODES * LENGTH):
        *_, infos = env.step(actions(N))
        got += [
            (float(i["episode"]["r"]), int(i["episode"]["l"]))
            for i in infos
            if "episode" in i
        ]
    return got


# ------------------------------------------------------------ plugrl-bridges


def _client(port: int):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]),
    )
    client = HERE.parent / "e50-boundary-transparency" / "gym_client.py"
    return subprocess.Popen(
        [sys.executable, str(client), "--port", str(port), "--num-envs", str(N),
         "--seed", "0", "--env-id", "counting_env:Counting-v0"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )  # fmt: skip


def _episode_log(path: pathlib.Path):
    with open(path, encoding="utf-8", newline="") as f:
        return [(float(r["return"]), int(r["length"])) for r in csv.DictReader(f)]


def bridges_sb3(port: int):
    """I: PlugRLVecEnv behind VecMonitor, and the server's episode log."""
    from stable_baselines3.common.vec_env import VecMonitor

    from plugrl_bridges.sb3 import PlugRLVecEnv

    log = pathlib.Path(tempfile.mkdtemp()) / "episodes.csv"
    client = _client(port)
    env = VecMonitor(
        PlugRLVecEnv(num_envs=N, action_space=Counting.action_space, port=port,
                     host="127.0.0.1", episode_log=str(log), timeout=60)
    )  # fmt: skip
    env.reset()
    got = []
    for _ in range(EPISODES * LENGTH):
        *_, infos = env.step(actions(N))
        got += [
            (float(i["episode"]["r"]), int(i["episode"]["l"]))
            for i in infos
            if "episode" in i
        ]
    env.close()
    client.wait(timeout=30)
    return {"VecMonitor": got, "episode log": _episode_log(log)}


def bridges_gym(port: int):
    """J: PlugRLVectorEnv's own infos["episode"], and the server's episode log."""
    from plugrl_bridges.gym import PlugRLVectorEnv

    log = pathlib.Path(tempfile.mkdtemp()) / "episodes.csv"
    client = _client(port)
    env = PlugRLVectorEnv(num_envs=N, action_space=Counting.action_space, port=port,
                          host="127.0.0.1", episode_log=str(log), timeout=60)  # fmt: skip
    env.reset()
    got = []
    for _ in range(EPISODES * LENGTH):
        *_, infos = env.step(actions(N))
        if "episode" in infos:
            got += masked(infos["episode"], infos["_episode"])
    env.close()
    client.wait(timeout=30)
    return {"infos": got, "episode log": _episode_log(log)}


# --------------------------------------------------------- plugrl-env-client


def _env_client_wrapper(src: pathlib.Path):
    path = src / "plugrl_env_client" / "utils" / "wrappers" / "episode_stats_wrapper.py"
    spec = importlib.util.spec_from_file_location("episode_stats_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VectorEpisodeStatsWrapper


class ResetByIndex:
    """What the env client's environments are to the wrapper: no autoreset,
    and `reset(options={"reset_indices": ...})` resets the envs named."""

    num_envs = N

    def __init__(self) -> None:
        self.envs = [Counting() for _ in range(N)]

    def reset(self, *, seed=None, options=None):
        indices = (options or {}).get("reset_indices")
        for i in range(N) if indices is None else indices:
            self.envs[i].reset()
        return np.zeros((N, 1), np.float32), {}

    def step(self, actions):
        out = [env.step(a) for env, a in zip(self.envs, actions)]
        return (
            np.stack([o[0] for o in out]),
            np.asarray([o[1] for o in out]),
            np.asarray([o[2] for o in out]),
            np.asarray([o[3] for o in out]),
            {},
        )


def env_client_runner(src: pathlib.Path):
    """K: the wrapper as rollout.py drives it: reset the envs that ended."""
    env = _env_client_wrapper(src)(ResetByIndex())
    env.reset(seed=0)
    got = []
    for _ in range(EPISODES * LENGTH):
        _, _, terminated, truncated, info = env.step(actions(N))
        if "episode" in info:
            got += [(float(info["episode"]["r"][i]), int(info["episode"]["l"][i]))
                    for i in np.flatnonzero(info["episode"]["mask"])]  # fmt: skip
        done = np.flatnonzero(np.logical_or(terminated, truncated))
        if done.size:
            env.reset(options={"reset_indices": done})
    return got


def env_client_no_resets(src: pathlib.Path):
    """Reported: the same wrapper over gymnasium's SyncVectorEnv, no explicit resets."""
    env = _env_client_wrapper(src)(gym.vector.SyncVectorEnv([Counting] * N))
    env.reset(seed=0)
    got = []
    for _ in range(steps_for(NEXT)):
        *_, info = env.step(actions(N))
        if "episode" in info:
            got += [(float(info["episode"]["r"][i]), int(info["episode"]["l"][i]))
                    for i in np.flatnonzero(info["episode"]["mask"])]  # fmt: skip
    return got


# ----------------------------------------------------------------------- main


def run(name: str, fn, expected):
    try:
        got = fn()
    except Exception as e:  # noqa: BLE001 - a raise is a result here
        got = f"raised {type(e).__name__}: {e}"
        tail = traceback.format_exc().strip().splitlines()[-3:]
        detail = "\n      ".join(tail)
    else:
        detail = ""
    results = got if isinstance(got, dict) else {"": got}
    ok = True
    for part, value in results.items():
        if isinstance(expected, str):
            match = isinstance(value, str) and value.startswith(expected)
        elif expected is None:
            match = None
        else:
            match = (
                not isinstance(value, str)
                and len(value) == len(expected)
                and all(
                    abs(r - er) < 1e-4 and n == en
                    for (r, n), (er, en) in zip(value, expected)
                )
            )
        ok = ok and match is not False
        label = f"{name} {part}".strip()
        print(f"{label}: {value}")
    if detail:
        print(f"      {detail}")
    verdict = (
        "reported"
        if expected is None
        else ("as predicted" if ok else "NOT as predicted")
    )
    print(f"   -> {verdict}\n")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8830)
    p.add_argument(
        "--env-client-src",
        default=os.environ.get(
            "PLUGRL_ENV_CLIENT_SRC",
            "D:/75128/Desktop/plugrl-work/plugrl-env-client/src",
        ),
    )
    a = p.parse_args()
    src = pathlib.Path(a.env_client_src)
    print(f"gymnasium {gym.__version__}, numpy {np.__version__}\n")
    later = [(15.0, 5)] * N + [(14.0, 4)] * (N * (EPISODES - 1))
    verdicts = {
        "P1": run("A vector wrapper, NEXT_STEP", lambda: vector_wrapper(NEXT), RIGHT),
        "P2": run("B vector wrapper, SAME_STEP", lambda: vector_wrapper(SAME), later),
        "P3": run("C per-env wrapper, NEXT_STEP", lambda: per_env_wrapper(NEXT), RIGHT)
        & run("D per-env wrapper, SAME_STEP", lambda: per_env_wrapper(SAME), RIGHT),
        "P4": run("E CleanRL logging, NEXT_STEP", lambda: cleanrl_logging(NEXT), [])
        & run(
            "F CleanRL logging, SAME_STEP",
            lambda: cleanrl_logging(SAME),
            "raised TypeError",
        ),
        "P5": run("G SB3 VecMonitor", sb3_vec_monitor, RIGHT)
        & run("H SB3 Monitor", sb3_monitor, RIGHT)
        & run("I plugrl-bridges sb3", lambda: bridges_sb3(a.port), RIGHT)
        & run("J plugrl-bridges gym", lambda: bridges_gym(a.port + 1), RIGHT)
        & run(
            "K env-client wrapper, its runner", lambda: env_client_runner(src), RIGHT
        ),
    }
    run(
        "reported: env-client wrapper, SyncVectorEnv, no resets",
        lambda: env_client_no_resets(src),
        None,
    )
    for name, ok in verdicts.items():
        print(f"{name}: {'holds' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
