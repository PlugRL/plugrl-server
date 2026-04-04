from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from plugrl_server.policy.state import NumpyState, PolicyTrainState


def _require_array(value: NumpyState, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value)!r}.")
    return value


def _require_mapping(value: NumpyState, name: str) -> Mapping[str, NumpyState]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)!r}.")
    return value


@dataclasses.dataclass(frozen=True)
class DPPOObsTrainState:
    x: np.ndarray
    t: np.ndarray
    cond: NumpyState


@dataclasses.dataclass(frozen=True)
class DPPOTrainState:
    obs: DPPOObsTrainState
    action: np.ndarray
    logprob: np.ndarray
    entropy: np.ndarray
    value: np.ndarray


def as_dppo_train_state(train_state: PolicyTrainState) -> DPPOTrainState:
    if train_state is None:
        raise ValueError("DPPO train_state must not be None.")

    obs_mapping = _require_mapping(train_state["obs"], "train_state['obs']")
    cond = obs_mapping["cond"]
    if not isinstance(cond, (np.ndarray, Mapping)):
        raise TypeError(
            f"train_state['obs']['cond'] must be a numpy tree, got {type(cond)!r}."
        )

    return DPPOTrainState(
        obs=DPPOObsTrainState(
            x=_require_array(obs_mapping["x"], "train_state['obs']['x']"),
            t=_require_array(obs_mapping["t"], "train_state['obs']['t']"),
            cond=cond,
        ),
        action=_require_array(train_state["action"], "train_state['action']"),
        logprob=_require_array(train_state["logprob"], "train_state['logprob']"),
        entropy=_require_array(train_state["entropy"], "train_state['entropy']"),
        value=_require_array(train_state["value"], "train_state['value']"),
    )
