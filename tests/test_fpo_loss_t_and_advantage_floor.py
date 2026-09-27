"""Two FPO options for testing why its first iteration destroys pi0.5 (E32).

* `loss_t_beta` - the flow times at which FPO evaluates its CFM loss. By
  default they are the policy's own denoising grid, uniformly; pi0 was
  trained at `0.999 * Beta(1.5, 1) + 0.001`, weighted towards noise. With the
  option set, FPO samples its times the way pi0's training did.
* `advantage_floor` - with 0, a sample whose advantage is negative carries no
  gradient: the update only pulls towards what went well, never pushes away
  from what went badly.

Both default to off, and off is FPO's behaviour until now.
"""

from __future__ import annotations

import numpy as np
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_buffer import FPOBuffer
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.algorithm.fpo.utils import compute_cfm_loss
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

B, S = 32, 4


def _policy() -> FPOPolicy:
    torch.manual_seed(0)
    return FPOPolicy(FPOPolicyConfig(device="cpu"))


def _buffer(policy: FPOPolicy, **kwargs) -> FPOBuffer:
    algo = FPOAlgorithm(FPOAlgoConfig(buffer_size=B, batch_size=B), policy)
    return FPOBuffer(
        buffer_size=B,
        example_train_state=algo.example_train_state(batch_size=1),
        n_samples_per_action=S,
        **kwargs,
    )


def test_both_options_default_to_off():
    config = FPOAlgoConfig()
    assert config.loss_t_beta is None
    assert config.advantage_floor is None


def test_by_default_the_times_are_the_policys_grid():
    policy = _policy()
    t = _buffer(policy)._sample_loss_t(policy, batch_size=4096)
    grid = set(np.round(policy._get_timesteps().numpy(), 6).tolist())
    assert set(np.round(t.numpy().ravel(), 6).tolist()) <= grid


def test_a_beta_draws_the_times_as_pi0_was_trained():
    """0.999 * Beta(1.5, 1) + 0.001: mean 0.6, off the grid, inside (0, 1]."""
    policy = _policy()
    torch.manual_seed(1)
    t = _buffer(policy, loss_t_beta=(1.5, 1.0))._sample_loss_t(policy, batch_size=8192)
    values = t.numpy().ravel()
    assert t.shape == (8192, S, 1)
    assert values.min() >= 0.001 and values.max() <= 1.0
    assert abs(values.mean() - (0.001 + 0.999 * 0.6)) < 0.01
    grid = set(np.round(policy._get_timesteps().numpy(), 6).tolist())
    assert len(set(np.round(values, 6).tolist()) - grid) > 1000


def _policy_gradient(advantage: torch.Tensor, **overrides) -> float:
    policy = _policy()
    algo = FPOAlgorithm(FPOAlgoConfig(buffer_size=B, batch_size=B, **overrides), policy)
    obs = torch.randn(B, 17)
    action = torch.randn(B, 1, 6)
    loss_eps = torch.randn(B, S, 1, 6)
    loss_t = torch.rand(B, S, 1)
    with torch.no_grad():
        initial = compute_cfm_loss(
            policy, obs, action, output_mode=algo.config.output_mode,
            loss_eps=loss_eps, loss_t=loss_t, obs_cache=policy.build_obs_cache(obs),
        )  # fmt: skip
    zeros = torch.zeros(B)
    policy.actor.zero_grad()
    loss, *_ = algo._compute_loss(
        obs, action, zeros, zeros, advantage, zeros, loss_eps, loss_t, initial
    )
    loss.backward()
    return sum(
        float(p.grad.pow(2).sum())
        for p in policy.actor.parameters()
        if p.grad is not None
    )


def test_a_floor_of_zero_removes_every_negative_sample_from_the_gradient():
    negative = -torch.linspace(0.5, 2.0, B)
    assert _policy_gradient(negative) > 0.0
    assert _policy_gradient(negative, advantage_floor=0.0) == 0.0


def test_a_floor_of_zero_keeps_the_positive_samples():
    positive = torch.linspace(0.5, 2.0, B)
    assert _policy_gradient(positive, advantage_floor=0.0) == _policy_gradient(positive)


def test_both_parse_from_the_command_line():
    """E32 passes them as `--algo.loss-t-beta 1.5 1.0` and `--algo.advantage-floor 0`."""
    import tyro

    config = tyro.cli(
        FPOAlgoConfig, args=["--loss-t-beta", "1.5", "1.0", "--advantage-floor", "0"]
    )
    assert config.loss_t_beta == (1.5, 1.0)
    assert config.advantage_floor == 0.0
