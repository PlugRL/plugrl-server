"""E55: E54's train.py, with the client's step ledger, and a client elsewhere.

Two changes from E54's:
- the client is also given --ledger, and writes ledger.npz beside the run;
- with --external-client, no client is started here: the bridge listens on
  every address, and a client started on another machine (cross.sh) connects
  to it. The run's client_rc is then the other machine's to record.
Run it with --trace: ledger.py compares trace.npz, what the trainer sent and
received, with ledger.npz, what the environments took and gave.

E54's docstring:

E54: SB3's PPO behind the bridge, with one fault at the boundary.

    python train.py --env pendulum --fault reward:stale --dose 0.01 --seed 0 \
        --out runs/... [--port 8854] [--trace]

E50's bridge arm, unchanged but for the fault: 16 environments in
faulty_client.py, another process, on one connection, behind plugrl-bridges'
PlugRLVecEnv; then VecMonitor, then PPO, one torch thread, CPU.
- pendulum: Pendulum-v1, 102,400 steps, rl-baselines3-zoo's settings, E50's.
- halfcheetah: HalfCheetah-v5, 409,600 steps, SB3's defaults, E50's.

Every fault but two is the client's (faulty_client.py lists them). The two
here are in the trainer's log, and leave what PPO gets as it came:
- log:drop-first: VecMonitor's logged return leaves out the episode's first
  reward (the kind of fault E51 found in Gymnasium's RecordEpisodeStatistics);
- log:short: VecMonitor's logged length is one step short (E51's other half).
Their doses count episodes: "one" is the 256th logged, a number p each with
probability p, from a generator seeded 54.

With --trace, a pass-through wrapper between the bridge and VecMonitor keeps
what crossed the wire as the trainer saw it, step by step: the actions sent,
and the rewards, episode ends, final observations and next observations
received. trace.npz holds them, for localise.py.

The run directory gets result.json (the final weights' SHA-256, the time
model.learn() took, and the error if the run raised one), monitor.csv (the
trainer's log), episodes.csv (the bridge's log, counted where the rewards
arrive on the trainer's side of the wire), env_record.csv and fault.json (the
client's), client.log, model.zip (the final policy, saved after its hash is
taken, for rendering), and trace.npz with --trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from faulty_client import FAULTS as CLIENT_FAULTS  # noqa: E402  (E55's copy)

NUM_ENVS = 16
ENVS = {
    "pendulum": dict(
        id="Pendulum-v1",
        total=102_400,
        action=(1, 2.0),
        # plugrl-bridges examples/sb3/ppo.py, preset "pendulum"; E50's.
        ppo=dict(
            n_steps=256,
            gamma=0.9,
            gae_lambda=0.95,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
        ),
    ),
    "halfcheetah": dict(
        id="HalfCheetah-v5",
        total=409_600,
        action=(6, 1.0),
        ppo=dict(n_steps=256),
    ),
}
LOG_FAULTS = ("log:drop-first", "log:short")
FAULTS = CLIENT_FAULTS + LOG_FAULTS


class FaultyLogMonitor(VecMonitor):
    """VecMonitor with one fault in the episodes it logs; PPO's data is untouched.

    step_wait is SB3 2.9.0's VecMonitor.step_wait, but for the two marked lines.
    """

    def __init__(self, venv, filename: str, fault: str, dose: str) -> None:
        super().__init__(venv, filename=filename)
        self.fault, self.dose = fault, dose
        self.rng = np.random.default_rng(54)
        self.first_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.logged = 0
        self.cells = 0

    def _hit(self) -> bool:
        k, u = self.logged, self.rng.random()
        self.logged += 1
        return k == 255 if self.dose == "one" else u < float(self.dose)

    def step_wait(self):
        starting = self.episode_lengths == 0
        obs, rewards, dones, infos = self.venv.step_wait()
        self.first_reward[starting] = rewards[starting]
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                if self._hit():  # the fault
                    self.cells += 1
                    if self.fault == "log:drop-first":
                        episode_return = episode_return - self.first_reward[i]
                    else:
                        episode_length = episode_length - 1
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos


class WireTrace(VecEnvWrapper):
    """What crossed the wire, as the trainer saw it; passes everything through."""

    def __init__(self, venv) -> None:
        super().__init__(venv)
        self.actions, self.rewards, self.ends, self.truncs = [], [], [], []
        self.finals, self.nexts = [], []
        self.first_obs = None

    def reset(self):
        obs = self.venv.reset()
        if self.first_obs is None:
            self.first_obs = np.array(obs, copy=True)
        return obs

    def step_async(self, actions) -> None:
        self.actions.append(np.array(actions, dtype=np.float32, copy=True))
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        final = np.zeros_like(obs)
        trunc = np.zeros(len(dones), dtype=np.bool_)
        for i in np.flatnonzero(dones):
            final[i] = infos[i]["terminal_observation"]
            trunc[i] = infos[i]["TimeLimit.truncated"]
        self.rewards.append(np.array(rewards, copy=True))
        self.ends.append(np.array(dones, copy=True))
        self.truncs.append(trunc)
        self.finals.append(final)
        self.nexts.append(np.array(obs, copy=True))
        return obs, rewards, dones, infos

    def save(self, path: pathlib.Path) -> None:
        np.savez_compressed(
            path,
            first_obs=self.first_obs,
            actions=np.stack(self.actions),
            rewards=np.stack(self.rewards),
            ends=np.stack(self.ends),
            truncs=np.stack(self.truncs),
            finals=np.stack(self.finals),
            nexts=np.stack(self.nexts),
        )


def weights_sha256(model: PPO) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(ENVS), required=True)
    p.add_argument("--fault", choices=FAULTS, required=True)
    p.add_argument("--dose", default="1.0")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8854)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--external-client", action="store_true")
    a = p.parse_args()
    spec = ENVS[a.env]
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    from plugrl_bridges.sb3 import PlugRLVecEnv

    dim, high = spec["action"]
    client_fault = a.fault if a.fault in CLIENT_FAULTS else "none"
    client = None
    if not a.external_client:
        client_log = open(out / "client.log", "w", encoding="utf-8")
        client = subprocess.Popen(
            [sys.executable, str(HERE / "faulty_client.py"), "--port", str(a.port),
             "--num-envs", str(NUM_ENVS), "--seed", str(a.seed), "--env-id", spec["id"],
             "--action-high", str(high), "--fault", client_fault, "--dose", a.dose,
             "--record", str(out / "env_record.csv"),
             "--ledger", str(out / "ledger.npz")],
            stdout=client_log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    env = PlugRLVecEnv(
        num_envs=NUM_ENVS,
        action_space=gym.spaces.Box(-high, high, (dim,), np.float32),
        port=a.port,
        host="0.0.0.0" if a.external_client else "127.0.0.1",
        timeout=60.0,
        episode_log=str(out / "episodes.csv"),
    )
    trace = WireTrace(env) if a.trace else None
    inner = trace if trace is not None else env
    if a.fault in LOG_FAULTS:
        env = FaultyLogMonitor(inner, str(out / "monitor.csv"), a.fault, a.dose)
    else:
        env = VecMonitor(inner, filename=str(out / "monitor.csv"))
    # PPO(seed=) seeds torch and numpy; the client resets env i with seed + i.
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, **spec["ppo"])
    result = {"env": a.env, "fault": a.fault, "dose": a.dose, "seed": a.seed}
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=spec["total"])
    except Exception as exc:  # a fault the bridge rejects ends the run here
        result["raised"] = f"{type(exc).__name__}: {exc}"
    result.update(
        weights_sha256=weights_sha256(model),
        learn_s=round(time.perf_counter() - start, 3),
        steps=model.num_timesteps,
        torch=torch.__version__,
    )
    if isinstance(env, FaultyLogMonitor):
        result["log_cells"] = env.cells
    model.save(out / "model.zip")
    if trace is not None:
        trace.save(out / "trace.npz")
    try:
        env.close()
    except Exception as exc:
        result["close_raised"] = f"{type(exc).__name__}: {exc}"
    if client is None:
        result["client_rc"] = "external"
    else:
        try:
            result["client_rc"] = client.wait(timeout=60)
        except subprocess.TimeoutExpired:
            client.kill()
            result["client_rc"] = "killed"
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
