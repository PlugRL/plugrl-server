"""`ppo dppo-square`: DPPO's Gaussian-policy PPO on robomimic square.

DPPO fine-tunes its released Gaussian MLP on square with
`cfg/robomimic/finetune/square/ft_ppo_gaussian_mlp.yaml` (irom-lab/dppo
cc7234ad). Where that differs from CleanRL's PPO, `ppo` gains an option, each
off by default:

  critic_learning_rate   a separate rate for the critic (1e-3; actor 1e-4)
  n_critic_warmup_itrs   iterations in which only the critic learns
  max_grad_norm = None   no gradient clipping
  reward_scaling_gamma   the discount of the running return that rewards are
                         scaled by (DPPO's RunningRewardScaler: 0.99), apart
                         from GAE's discount (0.999)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.ppo import ppo as ppo_module
from plugrl_server.algorithm.ppo.ppo import PPOAlgorithm
from plugrl_server.algorithm.ppo.ppo_buffer import PPOBuffer
from plugrl_server.algorithm.ppo.ppo_config import PPOAlgoConfig
from plugrl_server.algorithm.registration import REGISTERED_ALGO_CONFIGS
from test_gaussian_ppo import _algo, _collect, _iteration


def _state(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _moved(before, module) -> list[str]:
    return [k for k, v in module.state_dict().items() if not torch.equal(v, before[k])]


def test_the_variant_carries_dppos_square_settings():
    config = REGISTERED_ALGO_CONFIGS["ppo"]["dppo-square"]

    assert (config.learning_rate, config.critic_learning_rate) == (1e-4, 1e-3)
    assert config.anneal_lr is False
    assert (config.buffer_size, config.batch_size, config.update_epochs) == (
        20000,
        10000,
        10,
    )
    assert (config.gamma, config.gae_lambda) == (0.999, 0.95)
    assert (config.clip_coef, config.clip_vloss, config.vf_coef) == (0.01, False, 0.5)
    assert config.ent_coef == 0.0 and config.max_grad_norm is None
    assert config.target_kl == 1.0 and config.adam_eps == 1e-8
    assert config.normalize_rewards and config.reward_scaling_gamma == 0.99
    assert config.n_critic_warmup_itrs == 1


class TestTheOptimizer:
    def test_unset_it_is_cleanrls_one_adam(self):
        (group,) = _algo().optimizer.param_groups

        assert group["eps"] == 1e-5

    def test_a_critic_rate_splits_actor_and_critic(self):
        algo = _algo(critic_learning_rate=1e-3, learning_rate=1e-4, adam_eps=1e-8)

        actor, critic = algo.optimizer.param_groups
        critic_ids = {id(p) for p in algo.policy.critic.parameters()}
        assert {id(p) for p in critic["params"]} == critic_ids
        assert {id(p) for p in actor["params"]} == {
            id(p) for p in algo.policy.parameters() if id(p) not in critic_ids
        }
        assert (actor["lr"], critic["lr"]) == (1e-4, 1e-3)
        assert actor["eps"] == critic["eps"] == 1e-8

    def test_annealing_scales_both_rates(self):
        algo = _algo(critic_learning_rate=1e-3, learning_rate=1e-4, train_itrs=4)
        rng = np.random.default_rng(0)
        _iteration(algo, rng)
        _iteration(algo, rng)

        actor, critic = algo.optimizer.param_groups
        assert actor["lr"] == pytest.approx(1e-4 * 0.75)
        assert critic["lr"] == pytest.approx(1e-3 * 0.75)


class TestTheCriticWarmup:
    def test_the_actor_does_not_move_during_it_and_the_critic_does(self):
        algo = _algo(n_critic_warmup_itrs=1, critic_learning_rate=1e-3)
        critic_ids = {id(p) for p in algo.policy.critic.parameters()}
        before = {n: p.detach().clone() for n, p in algo.policy.named_parameters()}

        _iteration(algo, np.random.default_rng(0))

        for name, p in algo.policy.named_parameters():
            moved = not torch.equal(p.detach(), before[name])
            assert moved == (id(p) in critic_ids), name

    def test_after_it_the_actor_moves(self):
        algo = _algo(n_critic_warmup_itrs=1)
        rng = np.random.default_rng(0)
        _iteration(algo, rng)
        before = _state(algo.policy.actor_mean)

        _iteration(algo, rng)

        assert _moved(before, algo.policy.actor_mean) != []


def test_no_clipping_when_the_norm_is_none(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ppo_module.nn.utils, "clip_grad_norm_", lambda *a, **k: calls.append(1)
    )

    _, metrics = _iteration(_algo(max_grad_norm=None), np.random.default_rng(0))

    assert calls == []
    assert np.isfinite(metrics["train"]["max_grad_norm"])


def test_rewards_are_scaled_by_a_return_with_its_own_discount():
    algo = _algo()
    buffer = PPOBuffer(
        buffer_size=8,
        example_train_state=algo.example_train_state(batch_size=1),
        gamma=0.999,
        reward_scaling_gamma=0.5,
    )
    one = algo.example_train_state(batch_size=1)
    node = (-1, "")
    for _ in range(3):
        node = buffer.add_frame(
            prev_node=node, train_state=one, reward=1.0, terminated=False,
            truncated=False, last_value=None, next_terminated=False,
            next_truncated=False,
        )  # fmt: skip

    np.testing.assert_allclose(buffer.rets[:3], [1.0, 1.5, 1.75])


def test_the_config_default_keeps_the_rets_discount_at_gamma():
    assert PPOAlgoConfig().reward_scaling_gamma is None


def test_it_drives_dppos_gaussian_policy_end_to_end():
    pytest.importorskip("dppo")
    from plugrl_server.policy.dppo.dppo_gaussian_policy import (
        DPPOGaussianPolicy,
        DPPOGaussianPolicyConfig,
    )
    import test_gaussian_ppo as harness

    torch.manual_seed(0)
    policy = DPPOGaussianPolicy(DPPOGaussianPolicyConfig(device="cpu"))
    config = REGISTERED_ALGO_CONFIGS["ppo"]["dppo-square"]
    algo = PPOAlgorithm(
        PPOAlgoConfig(**{**config.__dict__, "buffer_size": 64, "batch_size": 32,
                         "n_critic_warmup_itrs": 0}),
        policy,
    )  # fmt: skip
    algo.init_optimizers()
    keys = policy.config.state_keys
    sizes = (3, 4, 2, 14)

    def obs(envs, rng):
        return {
            "states": {
                k: rng.uniform(-0.3, 0.3, size=(envs, n)).astype(np.float32)
                for k, n in zip(keys, sizes)
            }
        }

    original = harness._obs
    harness._obs = obs
    try:
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        _collect(algo, np.random.default_rng(0), reward_fn=lambda a: float(a.mean()))
        algo.pre_learn()
        _, metrics = algo.learn()
        algo.post_learn()
    finally:
        harness._obs = original

    moved = [n for n, p in policy.named_parameters() if not torch.equal(p, before[n])]
    assert "network.logvar" in moved
    assert any(n.startswith("network.mlp_mean") for n in moved)
    assert any(n.startswith("critic.") for n in moved)
    assert np.isfinite(metrics["losses"]["policy_loss"])
