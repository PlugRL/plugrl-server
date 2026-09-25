"""Gradient accumulation has to average over what accumulated, not over a plan.

`learn_impl` divided every minibatch's loss by `grad_accum_steps` and then
stepped whenever a window closed - which is correct only when a full window
accumulates. DPPO's loop flushes at the end of every epoch and breaks early on
`target_kl`, so partial windows are the normal case.

E17 measured what that costs. Its buffer was 4,096 with `batch_size` 2,048, so
each epoch had **two** minibatches, and `grad_accum_steps` defaulted to 8: two
gradients, each divided by eight, applied as one step. A quarter of the
gradient it meant to apply, for a hundred iterations, while
`train/actor_max_grad_norm` sat at 5.1e-04 and the return did not move.

The division now happens at the step, over the number of backward passes that
actually reached it.
"""

from __future__ import annotations

import torch

from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
from plugrl_server.algorithm.train_utils import optimizer_step_if_ready
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig


def _param(value: float = 0.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((4,), value))


def _with_grad(parameter: torch.nn.Parameter, grad: float) -> torch.nn.Parameter:
    parameter.grad = torch.full_like(parameter, grad)
    return parameter


class TestTheHelper:
    def test_it_averages_over_the_accumulated_count(self):
        """Three backward passes of 3.0 each must step on 3.0, not 9.0."""
        parameter = _with_grad(_param(), 3.0)
        optimizer = torch.optim.SGD([parameter], lr=1.0)

        norm = optimizer_step_if_ready(
            optimizer=optimizer,
            parameters=[parameter],
            accum_steps=3,
            grad_accum_steps=8,
            force=True,
            accumulated=3,
        )

        # SGD at lr 1 moves by the gradient, so the parameter says what the
        # optimizer saw: 3.0/3 = 1.0 per element.
        assert torch.allclose(parameter.detach(), torch.full((4,), -1.0))
        assert norm == torch.linalg.vector_norm(torch.full((4,), 1.0)).item()

    def test_one_accumulated_pass_is_untouched(self):
        parameter = _with_grad(_param(), 3.0)
        optimizer = torch.optim.SGD([parameter], lr=1.0)

        optimizer_step_if_ready(
            optimizer=optimizer,
            parameters=[parameter],
            accum_steps=1,
            grad_accum_steps=8,
            force=True,
            accumulated=1,
        )

        assert torch.allclose(parameter.detach(), torch.full((4,), -3.0))

    def test_without_the_argument_nothing_is_scaled(self):
        """A caller that already scaled its own loss is unaffected."""
        parameter = _with_grad(_param(), 3.0)
        optimizer = torch.optim.SGD([parameter], lr=1.0)

        optimizer_step_if_ready(
            optimizer=optimizer,
            parameters=[parameter],
            accum_steps=1,
            grad_accum_steps=1,
            force=True,
        )

        assert torch.allclose(parameter.detach(), torch.full((4,), -3.0))

    def test_it_does_not_step_mid_window(self):
        parameter = _with_grad(_param(), 3.0)
        optimizer = torch.optim.SGD([parameter], lr=1.0)

        norm = optimizer_step_if_ready(
            optimizer=optimizer,
            parameters=[parameter],
            accum_steps=1,
            grad_accum_steps=4,
            accumulated=1,
        )

        assert norm is None
        assert torch.allclose(parameter.detach(), torch.zeros(4))
        assert parameter.grad is not None, "the gradient must survive to accumulate"

    def test_the_reported_norm_is_the_averaged_gradient(self):
        """Not the summed one. E17's 5.1e-04 was a quarter of a real number."""
        parameter = _with_grad(_param(), 8.0)
        optimizer = torch.optim.SGD([parameter], lr=0.0)

        norm = optimizer_step_if_ready(
            optimizer=optimizer,
            parameters=[parameter],
            accum_steps=4,
            grad_accum_steps=4,
            accumulated=4,
        )

        assert norm == torch.linalg.vector_norm(torch.full((4,), 2.0)).item()


class TestTheCallerPassesItThrough:
    def _algo(self, **overrides) -> DPPOAlgorithm:
        config = DPPOAlgoConfig(
            buffer_size=64, train_itrs=20, batch_size=8, **overrides
        )
        algo = DPPOAlgorithm(config, FPOPolicy(FPOPolicyConfig(device="cpu")))
        algo.init_optimizers()
        return algo

    def test_step_optimizers_averages_the_actor(self):
        algo = self._algo()
        for parameter in algo.policy.actor.parameters():
            parameter.grad = torch.full_like(parameter, 2.0)
        before = [p.detach().clone() for p in algo.policy.actor.parameters()]

        actor_norm, critic_norm = algo._step_optimizers(
            accum_steps=2,
            grad_accum_steps=8,
            actor_enabled=True,
            force=True,
            accumulated=2,
        )

        assert actor_norm is not None and critic_norm is not None
        moved = [
            (p.detach() - b).abs().max().item()
            for p, b in zip(algo.policy.actor.parameters(), before)
        ]
        assert max(moved) > 0, "the actor did not move at all"

    def test_the_actor_is_left_alone_during_a_warmup(self):
        """The control: `accumulated` must not smuggle a step past the warmup."""
        algo = self._algo(n_critic_warmup_itrs=5)
        for parameter in algo.policy.actor.parameters():
            parameter.grad = torch.full_like(parameter, 2.0)
        before = [p.detach().clone() for p in algo.policy.actor.parameters()]

        actor_norm, critic_norm = algo._step_optimizers(
            accum_steps=2,
            grad_accum_steps=8,
            actor_enabled=False,
            force=True,
            accumulated=2,
        )

        assert actor_norm is None
        assert critic_norm is not None, "the critic still learns during a warmup"
        for parameter, was in zip(algo.policy.actor.parameters(), before):
            assert torch.equal(parameter.detach(), was)
