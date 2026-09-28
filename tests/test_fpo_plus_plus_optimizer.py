"""FPO++'s optimizer, gradient clipping, advantage normalisation and CFM loss, as options.

E37 fine-tuned a behaviour-cloned `fpo-policy` on robomimic square with
FPO++'s policy loss and watched it fall from 0.52 to 0.26-0.39, while DPPO
lifted the same clone to 0.78-0.80. Its critic never fit the returns. What
E37 called "FPO++'s settings" was only the policy loss. FPO++'s square
fine-tuning (amazon-far/fpo-control, `manipulation_experiments/
finetune_online_rl.py`) also has:

  two AdamW optimizers   the actor at 1e-5, betas (0.9, 0.99); the critic at
                         1e-4; both eps 1e-5, weight decay 1e-6. E37 gave the
                         random critic the actor's 1e-5.
  gradient clipping      actor and critic separately, at 25 on square.
  advantages             normalised per minibatch, not over the buffer.
  a Huber CFM error      d^2 within delta = 1, 2 delta |d| - delta^2 beyond,
                         before the mean over dimensions and the sum over
                         steps.

Each is an option here, and every option defaults to off, which is FPO's
behaviour until now.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.fpo import fpo as fpo_module
from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.algorithm.fpo.utils import ChunkReduction, compute_cfm_loss
from test_fpo_chunk_loss_per_sample_ratio import _ZeroVelocity, _inputs
from test_fpo_tree_observations import (
    TreeObsFlowPolicy,
    _collect_and_learn,
    _config,
    _tree_env_obs,
)

FPO_PLUS_PLUS = dict(
    learning_rate=1e-5,
    critic_learning_rate=1e-4,
    adam_eps=1e-5,
    weight_decay=1e-6,
    actor_adam_beta2=0.99,
)


def _algo(**overrides) -> FPOAlgorithm:
    torch.manual_seed(0)
    np.random.seed(0)
    config = _config()
    for key, value in overrides.items():
        setattr(config, key, value)
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
    algo.init_optimizers()
    return algo


def _state(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _moved(before: dict[str, torch.Tensor], module) -> list[str]:
    return [k for k, v in module.state_dict().items() if not torch.equal(v, before[k])]


class TestTheOptimizer:
    def test_unset_it_is_the_one_adam_fpo_always_had(self):
        algo = _algo()

        assert type(algo.optimizer) is torch.optim.Adam
        (group,) = algo.optimizer.param_groups
        assert group["lr"] == algo.config.learning_rate
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0

    def test_fpo_plus_plus_gives_actor_and_critic_their_own_adamw_groups(self):
        algo = _algo(**FPO_PLUS_PLUS)

        assert isinstance(algo.optimizer, torch.optim.AdamW)
        actor, critic = algo.optimizer.param_groups
        assert {id(p) for p in actor["params"]} == {
            id(p) for p in algo.policy.actor.parameters()
        }
        assert {id(p) for p in critic["params"]} == {
            id(p) for p in algo.policy.critic.parameters()
        }
        assert (actor["lr"], actor["betas"]) == (1e-5, (0.9, 0.99))
        assert (critic["lr"], critic["betas"]) == (1e-4, (0.9, 0.999))
        for group in (actor, critic):
            assert (group["eps"], group["weight_decay"]) == (1e-5, 1e-6)

    def test_a_critic_rate_alone_keeps_the_actor_at_the_shared_rate(self):
        algo = _algo(critic_learning_rate=1e-3)

        actor, critic = algo.optimizer.param_groups
        assert (actor["lr"], critic["lr"]) == (algo.config.learning_rate, 1e-3)

    def test_a_warmup_leaves_the_actor_exactly_as_it_was_despite_weight_decay(self):
        """AdamW decays every parameter it steps, gradient or not. So the actor's
        group is stepped at a rate of zero during the warmup, not only with its
        gradients zeroed."""
        algo = _algo(n_critic_warmup_itrs=1, **FPO_PLUS_PLUS)
        actor, critic = _state(algo.policy.actor), _state(algo.policy.critic)

        _collect_and_learn(algo, _tree_env_obs)

        assert _moved(actor, algo.policy.actor) == []
        assert _moved(critic, algo.policy.critic) != []
        assert algo.optimizer.param_groups[0]["lr"] == 1e-5, "the rate was not restored"

    def test_after_the_warmup_the_actor_moves(self):
        algo = _algo(n_critic_warmup_itrs=1, **FPO_PLUS_PLUS)
        _collect_and_learn(algo, _tree_env_obs)
        actor = _state(algo.policy.actor)

        _collect_and_learn(algo, _tree_env_obs)

        assert _moved(actor, algo.policy.actor) != []

    def test_a_checkpoint_round_trips_both_groups(self):
        algo = _algo(**FPO_PLUS_PLUS)
        _collect_and_learn(algo, _tree_env_obs)
        checkpoint = algo.create_checkpoint()

        resumed = _algo(**FPO_PLUS_PLUS)
        resumed.load_checkpoint(checkpoint)

        assert [g["lr"] for g in resumed.optimizer.param_groups] == [1e-5, 1e-4]
        for a, b in zip(algo.optimizer.state_dict()["state"].values(),
                        resumed.optimizer.state_dict()["state"].values()):  # fmt: skip
            assert torch.equal(a["exp_avg"], b["exp_avg"])


class TestGradientClipping:
    def test_off_by_default(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            fpo_module.torch.nn.utils,
            "clip_grad_norm_",
            lambda params, max_norm, *a, **k: calls.append(max_norm),
        )
        _collect_and_learn(_algo(), _tree_env_obs)

        assert calls == []

    def test_actor_and_critic_are_clipped_separately(self, monkeypatch):
        calls = []
        real = torch.nn.utils.clip_grad_norm_

        def recording(params, max_norm, *args, **kwargs):
            params = list(params)
            calls.append((len(params), max_norm))
            return real(params, max_norm, *args, **kwargs)

        monkeypatch.setattr(fpo_module.torch.nn.utils, "clip_grad_norm_", recording)
        algo = _algo(max_grad_norm=25.0)
        n_actor = len(list(algo.policy.actor.parameters()))
        n_critic = len(list(algo.policy.critic.parameters()))

        metrics = _collect_and_learn(algo, _tree_env_obs)

        steps = algo.config.num_updates_per_batch * (
            algo.config.buffer_size // algo.config.batch_size
        )
        assert calls == [(n_actor, 25.0), (n_critic, 25.0)] * steps
        assert np.isfinite(metrics["fpo"]["actor_grad_norm_max"])
        assert np.isfinite(metrics["fpo"]["critic_grad_norm_max"])

    def test_a_tight_clip_bounds_the_step(self):
        """With a clip far below the gradient, every step is the clip's size."""
        algo = _algo(max_grad_norm=1e-6, learning_rate=1.0)
        seen = []
        real_step = algo.optimizer.step

        def step(*args, **kwargs):
            params = [p for g in algo.optimizer.param_groups for p in g["params"]]
            actor = params[: len(list(algo.policy.actor.parameters()))]
            norm = torch.linalg.vector_norm(
                torch.stack([p.grad.norm() for p in actor if p.grad is not None])
            )
            seen.append(float(norm))
            return real_step(*args, **kwargs)

        algo.optimizer.step = step
        _collect_and_learn(algo, _tree_env_obs)

        assert seen and max(seen) <= 1e-6 * (1 + 1e-4)


class TestAdvantageNormalisation:
    def test_per_minibatch_leaves_every_minibatch_at_unit_spread(self):
        metrics = _collect_and_learn(
            _algo(normalize_advantage_per_minibatch=True), _tree_env_obs
        )

        assert metrics["fpo"]["advantages_std"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["fpo"]["advantages_mean"] == pytest.approx(0.0, abs=1e-5)

    def test_over_the_buffer_a_minibatch_is_not_at_unit_spread(self):
        """The control: without it, the test above would pass either way."""
        metrics = _collect_and_learn(_algo(), _tree_env_obs)

        assert metrics["fpo"]["advantages_std"] != pytest.approx(1.0, abs=1e-5)

    def test_it_needs_normalisation_on(self):
        with pytest.raises(ValueError, match="normalize_advantage"):
            FPOAlgoConfig(
                normalize_advantage=False, normalize_advantage_per_minibatch=True
            )


class TestTheHuberError:
    def test_unset_the_error_is_squared(self):
        assert ChunkReduction().huber_delta is None

    def test_within_delta_squared_beyond_it_linear_and_continuous(self):
        action, loss_eps, loss_t = _inputs()
        delta = 0.5
        reduction = ChunkReduction(
            steps=4, dims=6, sum_over_steps=True, huber_delta=delta
        )

        got = compute_cfm_loss(
            _ZeroVelocity(), None, action, output_mode="u",
            loss_eps=loss_eps, loss_t=loss_t, reduction=reduction,
        )  # fmt: skip

        d = (action.unsqueeze(1) - loss_eps).abs()
        error = torch.where(d <= delta, d.pow(2), 2 * delta * d - delta**2)
        torch.testing.assert_close(got, error.mean(-1).sum(-1))
        # The two pieces meet at delta.
        assert 2 * delta * delta - delta**2 == pytest.approx(delta**2)

    def test_the_algorithm_reads_it_from_its_config(self):
        algo = _algo(
            output_mode="u", cfm_loss_steps=1, cfm_loss_dims=1,
            cfm_loss_sum_over_steps=True, cfm_loss_huber_delta=1.0,
        )  # fmt: skip

        assert algo.chunk_reduction == ChunkReduction(1, 1, True, 1.0)


class TestTheCriticIsVisible:
    """E37 could not say how much of the return its critic explained: FPO
    logged no explained variance, where DPPO always has."""

    @pytest.mark.parametrize("playground", [True, False])
    def test_learn_reports_the_critics_explained_variance(self, playground):
        metrics = _collect_and_learn(
            _algo(fpo_playground_trick=playground), _tree_env_obs
        )

        assert np.isfinite(metrics["rollout"]["explained_variance"])
