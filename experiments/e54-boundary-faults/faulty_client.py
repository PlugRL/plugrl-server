"""E54's env client: E50's gym_client.py, with one fault injected at the boundary.

    python faulty_client.py --port 8854 --seed 0 --fault reward:stale --dose 0.01 \
        --record runs/.../env_record.csv

With --fault none it sends exactly what E50's client sent. Otherwise it
changes one field of what crosses the wire, with one operator, in a share of
the cells (vector step, env) where that field crosses. The environments
themselves are untouched, and --record writes each episode's return and
length as the environments produced them, before any fault: the record a
trainer's log can be checked against.

Fields and operators. Each field takes the operators that make sense for it:
  obs        the observation in an infer (what the policy acts on)
  action     the action the client applies (as it decodes the reply)
  reward     the reward in a feedback
  final-obs  the observation an episode ended on, in its feedback (SB3
             bootstraps from it at a time limit)
    zero   the value replaced by zeros (a default filled in)
    stale  the value from the step before (a buffer reused)
    swap   the value from the next slot, (i + 1) mod n (a slot mixed up)
    f16    the value rounded through float16 (a lossy encoding)
    clip   (action only) clipped to half its range: [-1, 1] for Pendulum's
           [-2, 2] (a client that assumes another range)
  terminated:spurious  sent true where the episode did not end
  truncated:drop       a time limit is not sent: the client resets, the trainer
                       sees the episode go on
  truncated:as-term    a time limit is sent as a termination
  step:fill            the env is not stepped, and its feedback is filled in:
                       reward 0, the same observation, no episode end
(log:drop-first and log:short are the trainer's log's, in train.py.)

Faults the bridge rejects, at one cell; the run is expected to fail loudly:
  indices:reorder      a feedback's rows, env_indices included, in reverse order
  reward:short-array   a feedback with one reward too few

Controls, which change nothing training receives:
  none, delay-1ms (sleep 1 ms before each feedback), wire-float64
  (observations sent as float64; the values are Pendulum's float32s).

Doses: "one" is a single cell: in env 0, the first eligible one at or after
vector step 3,250 (no episode ends there) whose value the operator changes; a number p is each eligible cell with probability p, drawn
from a generator seeded 54 whatever the training seed, so every seed gets the
same cells. --record's sibling fault.json counts the cells whose value the
fault actually changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import gymnasium as gym
import numpy as np
import websockets.exceptions
import websockets.sync.client

from plugrl_protocol import msgpack_numpy

STOP_REASON = "plugrl-server-stop"
CONTROLS = ("none", "delay-1ms", "wire-float64")
VALUE_OPS = ("zero", "stale", "swap", "f16")
SILENT = (
    tuple(f"obs:{op}" for op in VALUE_OPS)
    + tuple(f"action:{op}" for op in VALUE_OPS + ("clip",))
    + tuple(f"reward:{op}" for op in VALUE_OPS)
    + tuple(f"final-obs:{op}" for op in VALUE_OPS)
    + ("terminated:spurious", "truncated:drop", "truncated:as-term", "step:fill")
)
LOUD = ("indices:reorder", "reward:short-array")
FAULTS = CONTROLS + SILENT + LOUD
ONE_STEP = 3_249  # 0-based: the 3,250th vector step, not an episode's last


def observation(obs: np.ndarray, fault: str) -> dict:
    if fault == "wire-float64":
        obs = obs.astype(np.float64)
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


class Selector:
    """Which cells a fault hits: the same ones for every training seed."""

    def __init__(self, n: int, dose: str) -> None:
        self.n = n
        self.dose = dose
        self.rng = np.random.default_rng(54)
        self.one_done = False
        self.hit_now = False
        self.u = np.ones(n)

    def draw(self) -> None:
        self.u = self.rng.random(self.n)  # one draw per vector step, always
        self.hit_now = False

    def hits(self, step: int, eligible: np.ndarray) -> np.ndarray:
        if self.dose == "one":
            out = np.zeros(self.n, dtype=np.bool_)
            if not self.one_done and step >= ONE_STEP and eligible[0]:
                out[0] = True
                self.one_done = True
                self.hit_now = True
            return out
        return eligible & (self.u < float(self.dose))


def apply(
    op: str,
    values: np.ndarray,
    hit: np.ndarray,
    previous: np.ndarray,
    bound: float = 1.0,
) -> np.ndarray:
    """Return values with the operator applied in the hit rows."""
    out = values.copy()
    if not hit.any():
        return out
    if op == "zero":
        out[hit] = 0
    elif op == "stale":
        out[hit] = previous[hit]
    elif op == "swap":
        out[hit] = np.roll(values, -1, axis=0)[hit]
    elif op == "f16":
        out[hit] = values[hit].astype(np.float16).astype(values.dtype)
    elif op == "clip":
        out[hit] = np.clip(values[hit], -bound / 2, bound / 2)
    else:
        raise ValueError(op)
    return out


def changed(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """The rows (envs) in which new differs from old."""
    diff = new != old
    return diff.reshape(len(diff), -1).any(axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8854)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--action-high", type=float, default=2.0)
    p.add_argument("--fault", choices=FAULTS, default="none")
    p.add_argument("--dose", default="1.0")
    p.add_argument("--record", required=True)
    a = p.parse_args()

    n, fault = a.num_envs, a.fault
    field, _, op = fault.partition(":")
    dose = "one" if fault in LOUD else a.dose
    select = Selector(n, dose)
    envs = [gym.make(a.env_id) for _ in range(n)]
    obs = np.stack([env.reset(seed=a.seed + i)[0] for i, env in enumerate(envs)])
    obs = obs.astype(np.float32)
    indices = np.arange(n, dtype=np.int64)
    step_ids = np.zeros(n, dtype=np.int64)
    packer = msgpack_numpy.Packer()
    steps = 0
    cells = 0
    every = np.ones(n, dtype=np.bool_)
    first: list[int] | None = None  # the first changed cell, [vector step, env]

    def tally(rows: np.ndarray) -> None:
        nonlocal cells, first
        if select.hit_now and not rows.any():
            select.one_done = (
                False  # "one": that cell's value did not change; try the next
            )
        if rows.any():
            cells += int(rows.sum())
            if first is None:
                first = [steps, int(np.flatnonzero(rows)[0])]

    # What the environments produced, before any fault.
    ep_return = np.zeros(n, dtype=np.float64)
    ep_length = np.zeros(n, dtype=np.int64)
    episodes = np.zeros(n, dtype=np.int64)
    record_path = pathlib.Path(a.record)
    record = open(record_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(record)
    writer.writerow(["step", "env", "episode", "return", "length"])

    prev_obs = obs.copy()
    prev_actions = None
    prev_rewards = np.zeros(n, dtype=np.float32)

    def finish(reason: str) -> int:
        record.close()
        (record_path.parent / "fault.json").write_text(
            json.dumps(
                {
                    "fault": fault,
                    "dose": dose,
                    "cells": cells,
                    "first": first,
                    "steps": steps,
                }
            )
            + "\n"
        )
        print(
            f"closed after {steps} steps: {reason!r}; {cells} cells changed", flush=True
        )
        return 0 if reason == STOP_REASON else 1

    ws = connect(f"ws://{a.host}:{a.port}", time.monotonic() + 120)
    with ws:
        msgpack_numpy.unpackb(ws.recv())  # the metadata; nothing in it is needed
        try:
            while True:
                select.draw()
                sent = obs
                if field == "obs":
                    sent = apply(op, obs, select.hits(steps, every), prev_obs)
                    tally(changed(sent, obs))
                ws.send(
                    packer.pack(
                        {
                            "message_type": "infer",
                            "data": observation(sent, fault),
                            "env_indices": indices,
                            "step_ids": step_ids.copy(),
                        }
                    )
                )
                prev_obs = obs.copy()
                reply = msgpack_numpy.unpackb(ws.recv())
                received = np.asarray(reply["data"]["action"])[0]  # [H, n, d] -> [n, d]
                actions = received
                if field == "action":
                    previous = (
                        np.zeros_like(received)
                        if prev_actions is None
                        else prev_actions
                    )
                    actions = apply(
                        op, received, select.hits(steps, every), previous, a.action_high
                    )
                    tally(changed(actions, received))
                prev_actions = received.copy()
                skip = np.zeros(n, dtype=np.bool_)
                if field == "step":
                    skip = select.hits(steps, every)
                    tally(skip)
                ended = np.empty_like(obs)
                after = np.empty_like(obs)
                rewards = np.zeros(n, dtype=np.float32)
                terminated = np.zeros(n, dtype=np.bool_)
                truncated = np.zeros(n, dtype=np.bool_)
                sent_ids = step_ids.copy()
                for i, env in enumerate(envs):
                    if skip[i]:
                        ended[i] = obs[i]
                        after[i] = obs[i]
                        step_ids[i] += 1
                        continue
                    o, r, te, tr, _ = env.step(actions[i])
                    ended[i] = o
                    rewards[i], terminated[i], truncated[i] = r, te, tr
                    ep_return[i] += float(rewards[i])
                    ep_length[i] += 1
                    if te or tr:
                        episodes[i] += 1
                        writer.writerow(
                            [
                                steps + 1,
                                i,
                                episodes[i],
                                f"{ep_return[i]:.6f}",
                                ep_length[i],
                            ]
                        )
                        ep_return[i] = 0.0
                        ep_length[i] = 0
                        after[i] = env.reset()[0]
                        step_ids[i] = 0
                    else:
                        after[i] = o
                        step_ids[i] += 1
                true_rewards = rewards
                ends = terminated | truncated
                final = ended
                if field == "reward" and op in VALUE_OPS:
                    rewards = apply(
                        op, true_rewards, select.hits(steps, every), prev_rewards
                    )
                    tally(changed(rewards, true_rewards))
                elif field == "final-obs":
                    final = apply(op, ended, select.hits(steps, ends), obs)
                    tally(changed(final, ended))
                elif fault == "terminated:spurious":
                    hit = select.hits(steps, ~terminated)
                    terminated = terminated | hit
                    tally(hit)
                elif fault == "truncated:drop":
                    hit = select.hits(steps, truncated)
                    truncated = truncated & ~hit
                    tally(hit)
                elif fault == "truncated:as-term":
                    hit = select.hits(steps, truncated)
                    terminated = terminated | hit
                    truncated = truncated & ~hit
                    tally(hit)
                prev_rewards = true_rewards.copy()
                if fault == "delay-1ms":
                    time.sleep(0.001)
                order = indices
                fields = {
                    "obs": observation(final, fault),
                    "rewards": rewards,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": {},
                }
                if fault in LOUD and select.hits(steps, every)[0]:
                    tally(indices == 0)
                    if fault == "indices:reorder":
                        order = indices[::-1].copy()
                        fields["obs"] = observation(final[::-1].copy(), fault)
                        for key in ("rewards", "terminated", "truncated"):
                            fields[key] = fields[key][::-1].copy()
                        sent_ids = sent_ids[::-1].copy()
                    else:
                        fields["rewards"] = rewards[:-1].copy()
                ws.send(
                    packer.pack(
                        {
                            "message_type": "feedback",
                            "env_indices": order,
                            "step_ids": sent_ids,
                            "data": fields,
                        }
                    )
                )
                obs = after
                steps += 1
        except websockets.exceptions.ConnectionClosed as closed:
            return finish(closed.rcvd.reason if closed.rcvd is not None else "")


if __name__ == "__main__":
    raise SystemExit(main())
