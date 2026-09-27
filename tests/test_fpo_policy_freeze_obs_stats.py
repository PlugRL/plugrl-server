"""`fpo-policy` can keep the observation statistics it was started from.

`fpo-policy` normalises observations by running statistics, updated with
every rollout buffer before each learn step. From scratch that is what it
needs. Started from a behaviour-cloned checkpoint it is not: the statistics
come from the demonstrations, the first rollouts differ, and updating them
changes what the cloned policy does before any learning has happened. On
robomimic square (E37's pilot) an iteration that trained only the critic
still moved the policy's own CFM loss far enough to put the mean ratio at
1.38, and the next iteration's success fell from 0.51 to 0.39. DPPO and
FPO++ both fine-tune with the normalisation of the demonstrations, fixed.
`freeze_obs_stats` keeps the statistics as they are.
"""

from __future__ import annotations

import torch

from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig


def _stats(policy: FPOPolicy) -> list[torch.Tensor]:
    return [
        policy.obs_stats_count.clone(),
        policy.obs_stats_mean.clone(),
        policy.obs_stats_var_sum.clone(),
        policy.obs_stats_std.clone(),
    ]


def _started(**overrides) -> FPOPolicy:
    torch.manual_seed(0)
    policy = FPOPolicy(FPOPolicyConfig(device="cpu", **overrides))
    # Statistics from "the demonstrations".
    policy.update_obs_stats(torch.randn(256, policy.config.obs_dim))
    return policy


def test_off_by_default():
    assert FPOPolicyConfig().freeze_obs_stats is False


def test_by_default_the_statistics_follow_new_data():
    policy = _started()
    before = _stats(policy)
    policy.update_obs_stats(torch.randn(64, policy.config.obs_dim) * 3 + 2)
    assert not all(torch.equal(a, b) for a, b in zip(before, _stats(policy)))


def test_frozen_statistics_stay_as_started():
    torch.manual_seed(0)
    source = _started()
    policy = FPOPolicy(FPOPolicyConfig(device="cpu", freeze_obs_stats=True))
    policy.load_state_dict(source.state_dict())
    before = _stats(policy)
    policy.update_obs_stats(torch.randn(64, policy.config.obs_dim) * 3 + 2)
    assert all(torch.equal(a, b) for a, b in zip(before, _stats(policy)))


def test_frozen_statistics_still_normalise():
    policy = _started(freeze_obs_stats=False)
    frozen = FPOPolicy(FPOPolicyConfig(device="cpu", freeze_obs_stats=True))
    frozen.load_state_dict(policy.state_dict())
    obs = torch.randn(8, policy.config.obs_dim)
    assert torch.equal(policy.build_obs_cache(obs), frozen.build_obs_cache(obs))


def test_it_parses_from_the_command_line():
    import tyro

    assert tyro.cli(FPOPolicyConfig, args=["--freeze-obs-stats"]).freeze_obs_stats
