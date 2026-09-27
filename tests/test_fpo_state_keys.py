"""fpo-policy reads its observation from named state keys, in order.

It read `states["obs"]` and nothing else, which is what the MuJoCo client
sends. The robomimic client sends one state per quantity - `robot0_eef_pos`,
`object` and the rest - so fpo-policy could not run on robomimic at all.
`state_keys` names the keys to concatenate; the default, `("obs",)`, is the
old behaviour exactly.

A key the observation lacks, or keys whose widths do not add up to
`obs_dim`, raise with what was there: the alternative is a shape error deep
in the network, or - worse - a policy silently fed the wrong slice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig


def _policy(**kw) -> FPOPolicy:
    return FPOPolicy(FPOPolicyConfig(device="cpu", **kw))


def test_the_default_reads_obs_as_before():
    policy = _policy(obs_dim=3, action_dim=2)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    got = policy.extract_model_obs_tensor({"states": {"obs": x, "other": x * 10}})

    assert torch.equal(got, torch.as_tensor(x))


def test_named_keys_are_concatenated_in_the_order_given():
    policy = _policy(obs_dim=5, action_dim=2, state_keys=("b", "a"))
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 3), 2.0, dtype=np.float32)

    got = policy.extract_model_obs_tensor({"states": {"a": a, "b": b, "c": a}})

    assert got.shape == (2, 5)
    assert torch.equal(got, torch.as_tensor(np.concatenate([b, a], axis=-1)))


def test_a_missing_key_names_what_was_there():
    policy = _policy(obs_dim=3, action_dim=2, state_keys=("object",))

    with pytest.raises(KeyError, match="robot0_eef_pos"):
        policy.extract_model_obs_tensor(
            {"states": {"robot0_eef_pos": np.zeros((1, 3), dtype=np.float32)}}
        )


def test_widths_that_do_not_add_up_to_obs_dim_raise():
    policy = _policy(obs_dim=4, action_dim=2, state_keys=("a", "b"))

    with pytest.raises(ValueError, match="obs_dim"):
        policy.extract_model_obs_tensor(
            {
                "states": {
                    "a": np.zeros((1, 2), dtype=np.float32),
                    "b": np.zeros((1, 3), dtype=np.float32),
                }
            }
        )
