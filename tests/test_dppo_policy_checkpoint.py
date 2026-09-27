"""`dppo-policy` starts from the weights DPPO itself fine-tunes, or refuses.

DPPO's pretraining saves two copies of the network, `model` and `ema`, and
DPPO's own `DiffusionModel` loads `ema` whenever a checkpoint has it. The
square checkpoint DPPO released carries both, 14.6% apart relative to
`model`'s norm. `dppo-policy` took `model`, with `strict=False`: it fine-tuned
other weights than DPPO does, and a checkpoint whose keys did not match would
have left the network at its random initialisation without a word.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("dppo")

from plugrl_server.policy.dppo.dppo_policy import DPPOPolicy, DPPOPolicyConfig  # noqa: E402


def _policy(checkpoint=None) -> DPPOPolicy:
    return DPPOPolicy(DPPOPolicyConfig(device="cpu", checkpoint_path=checkpoint))


def _shifted(state: dict, by: float) -> dict:
    return {k: v + by if v.is_floating_point() else v for k, v in state.items()}


def _same(actor: torch.nn.Module, state: dict) -> bool:
    mine = actor.state_dict()
    return mine.keys() == state.keys() and all(
        torch.equal(mine[k], state[k]) for k in state
    )


@pytest.fixture
def state() -> dict:
    return {k: v.clone() for k, v in _policy().actor.state_dict().items()}


def test_the_ema_weights_when_the_checkpoint_has_them(tmp_path, state):
    path = tmp_path / "state_8000.pt"
    torch.save(
        {"epoch": 8000, "model": _shifted(state, 1.0), "ema": _shifted(state, 2.0)},
        path,
    )
    assert _same(_policy(path).actor, _shifted(state, 2.0))


def test_the_model_weights_when_there_is_no_ema(tmp_path, state):
    path = tmp_path / "state_100.pt"
    torch.save({"epoch": 100, "model": _shifted(state, 1.0)}, path)
    assert _same(_policy(path).actor, _shifted(state, 1.0))


def test_a_checkpoint_missing_a_weight_is_refused(tmp_path, state):
    partial = dict(state)
    partial.pop(next(iter(partial)))
    path = tmp_path / "partial.pt"
    torch.save({"ema": partial}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        _policy(path)


def test_a_checkpoint_for_another_network_is_refused(tmp_path, state):
    path = tmp_path / "other.pt"
    torch.save({"ema": {**state, "network.not_this_one.weight": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _policy(path)
