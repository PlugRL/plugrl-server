"""DPPO's log-probability clamp, and what it does to a narrow policy.

DPPO's loss clamps every element's log-probability to [-5, 2] before forming
the ratio, as DPPO's reference does. With DPPO's own policy, whose every
denoising step has a standard deviation of at least `sampling_noise_level`
(0.1 in the MuJoCo variants), the upper bound never binds: the density of a
normal with deviation 0.1 is at most e^1.38. A flow policy's later steps are
narrower - `fpo-policy`'s run from 0.1 down to about 0.01 at the same level -
and there about half the elements land above 2. A clamped element carries
no gradient and a ratio of exactly 1.

The bounds are now configuration, defaulting to the reference's.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import tyro

from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

B = 64


def _algo(**overrides) -> DPPOAlgorithm:
    config = DPPOAlgoConfig(
        buffer_size=B,
        train_itrs=20,
        batch_size=B,
        sampling_noise_level=0.1,
        logprob_noise_level=0.1,
        **overrides,
    )
    torch.manual_seed(0)
    policy = FPOPolicy(FPOPolicyConfig(obs_dim=11, action_dim=3, device="cpu"))
    return DPPOAlgorithm(config, policy)


def _policy_gradient(algo: DPPOAlgorithm) -> tuple[float, float]:
    """The actor's gradient norm from the policy loss on one minibatch, and the
    fraction of log-probability elements above the upper bound."""
    obs = {
        "states": {
            "obs": np.random.default_rng(0).standard_normal((B, 11)).astype(np.float32)
        }
    }
    with torch.no_grad():
        _, state = algo.policy.get_action_and_runtime_state(
            obs, sampling_noise_level=algo.config.sampling_noise_level
        )
    model_obs = {"x": state.obs.x, "t": state.obs.t, "cond": state.obs.cond}
    advantage = torch.linspace(-1.0, 1.0, B)
    zeros = torch.zeros(B)

    algo.policy.actor.zero_grad()
    pg_loss, *_ = algo._compute_loss(
        model_obs, state.action, state.logprob, zeros, advantage, zeros
    )
    pg_loss.backward()
    norm = math.sqrt(
        sum(
            float(p.grad.pow(2).sum())
            for p in algo.policy.actor.parameters()
            if p.grad is not None
        )
    )
    high = float((state.logprob > algo.config.logprob_clamp_max).float().mean())
    return norm, high


def test_the_defaults_are_the_references_bounds():
    config = DPPOAlgoConfig()
    assert (config.logprob_clamp_min, config.logprob_clamp_max) == (-5.0, 2.0)


def test_the_upper_bound_binds_on_about_half_of_a_flow_policys_elements():
    """The finding itself, at E28's noise level."""
    _, high = _policy_gradient(_algo())
    assert 0.35 < high < 0.65


def test_a_clamped_element_carries_no_gradient():
    """Clamp everything and the policy loss cannot move the actor at all."""
    norm, high = _policy_gradient(
        _algo(logprob_clamp_min=-20.0, logprob_clamp_max=-10.0)
    )
    assert high > 0.99
    assert norm == 0.0


def test_lifting_the_bounds_restores_the_gradient_the_clamp_removed():
    clamped, _ = _policy_gradient(_algo())
    free, high = _policy_gradient(
        _algo(logprob_clamp_min=-math.inf, logprob_clamp_max=math.inf)
    )
    assert high == 0.0
    assert free > clamped > 0.0


def test_the_bounds_parse_from_the_command_line():
    """`inf` parses; `-inf` needs the equals form, or argparse reads it as a flag."""
    config = tyro.cli(
        DPPOAlgoConfig,
        args=["--logprob-clamp-min=-inf", "--logprob-clamp-max", "inf"],
    )
    assert config.logprob_clamp_min == -math.inf
    assert config.logprob_clamp_max == math.inf
