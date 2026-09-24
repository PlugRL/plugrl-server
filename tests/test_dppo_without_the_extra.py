"""DPPO runs on a plain install, and the CLI no longer offers what it cannot do.

README documented the state this replaces: with no `dppo` package installed,
`... fpo-policy default --help` still offered `{fpo,dummy,eval,dppo}`, because
the config module imported cleanly and registered its configs while the module
carrying the algorithm class was never imported at all. Selecting `dppo`
parsed, started up, printed its config, and died with
`KeyError: 'Algorithm dppo is not registered.'`

Two things caused that, and both are fixed here:

  * the algorithm reached into the external `dppo` package for two utilities -
    a running mean/variance and a learning-rate schedule - neither of which is
    DPPO-specific. Both are MIT and are now carried in `dppo/third_party/`.
  * `algorithm/__init__.py` imported the *config* modules and not the classes.

`dppo-policy` still needs `plugrl-server[dppo]`: it wraps DPPO's own
`DiffusionModel` and builds it through DPPO's hydra configs, which is a real
dependency on that project rather than on two utility classes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig, SchedulerConfig
from plugrl_server.algorithm.dppo.dppo_optimizer import build_adamw, build_scheduler
from plugrl_server.algorithm.dppo.dppo_scheduler import NoOpScheduler
from plugrl_server.algorithm.dppo.third_party.reward_scaling import RunningMeanStd
from plugrl_server.algorithm.registration import REGISTERED_ALGORITHMS
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig


class TestTheMenuMatchesWhatRuns:
    """The regression for the KeyError README used to describe."""

    def test_dppo_is_a_registered_algorithm(self):
        assert "dppo" in REGISTERED_ALGORITHMS

    def test_dppo_dist_is_a_registered_algorithm(self):
        assert "dppo-dist" in REGISTERED_ALGORITHMS

    def test_the_algorithm_constructs_against_a_flow_policy(self):
        """DPPO is typed against the diffusion base class, not its own policy.

        `BasePolicyGradientFlowPolicy` derives from it and overrides
        `_denoising_step`, so `fpo-policy` satisfies the interface. That is
        what makes DPPO usable as a control for FPO: the same policy, the same
        environment, two algorithms.
        """
        config = DPPOAlgoConfig(buffer_size=64, train_itrs=20, batch_size=8)
        algo = DPPOAlgorithm(config, FPOPolicy(FPOPolicyConfig(device="cpu")))
        algo.init_optimizers()

        assert algo.actor_optimizer is not None
        assert algo.critic_optimizer is not None

    def test_a_flow_policy_has_a_density_once_sampling_is_noisy(self):
        """Why DPPO can drive a flow policy at all.

        A deterministic flow has no tractable density - which is the reason
        FPO exists. Adding `sampling_noise_level` turns each denoising step
        into a Gaussian transition, and DPPO's per-step log-probability is
        that Gaussian's. With no noise the log-probability is exactly zero and
        DPPO would have nothing to form a ratio from.
        """
        policy = FPOPolicy(FPOPolicyConfig(device="cpu"))
        x = torch.zeros(4, policy.action_horizon, policy.action_dim)
        t = torch.full((4,), 0.5)
        cond = torch.zeros(4, 17)

        _, deterministic, _ = policy._denoising_step(x, t, cond)
        _, noisy, _ = policy._denoising_step(x, t, cond, sampling_noise_level=0.1)

        assert torch.equal(deterministic, torch.zeros_like(deterministic))
        assert not torch.equal(noisy, torch.zeros_like(noisy))


class TestRunningMeanStd:
    def test_one_batch_is_the_sample_variance(self):
        """Not the population variance.

        Upstream DPPO divides by `count - 1`, where the Baselines code it
        descends from divides by `count`. DPPO's published results were
        produced with `count - 1`, so this carries that. The initial
        pseudo-count of 1e-4 keeps it from being exact, which is why this is a
        tolerance and not an equality.
        """
        rng = np.random.default_rng(0)
        x = rng.normal(3.0, 2.0, size=5000)

        stats = RunningMeanStd(shape=())
        stats.update(x)

        assert stats.mean == pytest.approx(np.mean(x), rel=1e-3)
        assert stats.var == pytest.approx(np.var(x, ddof=1), rel=1e-3)
        assert stats.var != pytest.approx(np.var(x, ddof=0), rel=1e-9)

    def test_the_mean_is_exactly_order_invariant(self):
        rng = np.random.default_rng(1)
        first = rng.normal(0.0, 1.0, size=700)
        second = rng.normal(5.0, 3.0, size=300)

        incremental = RunningMeanStd(shape=())
        incremental.update(first)
        incremental.update(second)

        at_once = RunningMeanStd(shape=())
        at_once.update(np.concatenate([first, second]))

        assert incremental.mean == pytest.approx(at_once.mean, rel=1e-9)

    def test_the_variance_is_only_approximately_order_invariant(self):
        """A real property of the reference implementation, pinned on purpose.

        Welford's update is exact for M2, the sum of squared deviations. This
        one divides by `count - 1` to store `var`, then multiplies by `count`
        to recover M2 on the next update, so each step inflates M2 by roughly
        `1 + 1/count`. Two batches therefore do not give bit-identical
        variance to one batch of the concatenation - here about one part in
        ten thousand.

        Pinned rather than fixed. DPPO's published results were produced with
        this arithmetic, and a later "correction" to an exact Welford update
        would silently change what `use_normalized_rewards` does. The bound
        below is what makes such a change show up as a failing test instead.
        """
        rng = np.random.default_rng(1)
        first = rng.normal(0.0, 1.0, size=700)
        second = rng.normal(5.0, 3.0, size=300)

        incremental = RunningMeanStd(shape=())
        incremental.update(first)
        incremental.update(second)

        at_once = RunningMeanStd(shape=())
        at_once.update(np.concatenate([first, second]))

        assert incremental.var == pytest.approx(at_once.var, rel=1e-3)
        assert incremental.var != pytest.approx(at_once.var, rel=1e-9)

    def test_it_starts_at_mean_zero_variance_one(self):
        stats = RunningMeanStd(shape=())

        assert stats.mean == 0.0
        assert stats.var == 1.0


class TestScheduler:
    def _optimizer(self) -> torch.optim.AdamW:
        return build_adamw([torch.nn.Parameter(torch.zeros(2))], lr=1e-3, weight_decay=0)

    def test_no_config_means_no_schedule(self):
        scheduler = build_scheduler(
            self._optimizer(), scheduler_config=None, train_itrs=100, max_lr=1e-3
        )

        assert isinstance(scheduler, NoOpScheduler)
        assert scheduler.step() is None

    def test_a_run_shorter_than_its_warmup_is_refused_with_both_numbers(self):
        """The trap this replaces was a bare AssertionError inside a vendored file.

        A two-iteration smoke test against a ten-iteration warmup is a
        plausible first command, and the assertion it used to hit named
        neither number nor the file the numbers came from.
        """
        with pytest.raises(ValueError) as caught:
            build_scheduler(
                self._optimizer(),
                scheduler_config=SchedulerConfig(min_lr=1e-4, warmup_steps=10),
                train_itrs=2,
                max_lr=1e-3,
            )

        assert "10" in str(caught.value)
        assert "2" in str(caught.value)

    def test_it_warms_up_then_anneals(self):
        optimizer = self._optimizer()
        scheduler = build_scheduler(
            optimizer,
            scheduler_config=SchedulerConfig(min_lr=1e-5, warmup_steps=10),
            train_itrs=100,
            max_lr=1e-3,
        )

        def lr() -> float:
            return optimizer.param_groups[0]["lr"]

        assert lr() == pytest.approx(1e-5)
        for _ in range(10):
            scheduler.step()
        at_peak = lr()
        for _ in range(80):
            scheduler.step()

        assert at_peak == pytest.approx(1e-3, rel=1e-6)
        assert lr() < at_peak
        assert lr() >= 1e-5
