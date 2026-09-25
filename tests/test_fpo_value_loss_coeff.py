"""`value_loss_coeff` does nothing when the critic is disjoint under Adam.

Adam's update is `lr * m / (sqrt(v) + eps)`, which is invariant to scaling
the loss: multiply a gradient by a constant and both moments scale with it,
leaving the step unchanged. So a coefficient on the value loss only means
something for parameters that receive gradient from the *other* loss too.

In FPO's own configurations it often does not:

  the toy policy here    the critic is its own `nn.Linear`, disjoint from the
                         actor, so scaling the value loss is cancelled exactly
  pi0.5 as E14 ran it    `_get_value` reads `build_obs_cache`, which runs the
                         paligemma prefix with the expert's input set to
                         `None`, and `train_expert_only=True` sets
                         `requires_grad = False` on every paligemma
                         parameter - so the value gradient reaches the frozen
                         trunk and the disjoint critic and nothing else

Measured on the bandit: coefficients 0.25, 1.0 and 4.0 give byte-identical
results across five seeds. The apparent effect below about 0.05 is Adam's
epsilon becoming comparable to the scaled second moment, not the critic
learning differently.

This is not a behaviour change. Redefining the knob would silently alter what
it means for anyone reading it, and nothing can depend on it today because it
does nothing. It is written down so the next person to reach for it - as this
project did, using it as an intervention in an investigation - finds out here
instead of from a sweep that returns the same number three times.
"""

from __future__ import annotations

import numpy as np
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from test_fpo_tree_observations import (
    TreeObsFlowPolicy,
    _collect_and_learn,
    _config,
    _tree_env_obs,
)


def _run(coeff: float, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = _config()
    config.value_loss_coeff = coeff
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
    algo.init_optimizers()
    _collect_and_learn(algo, _tree_env_obs)
    return [p.detach().clone() for p in algo.policy.parameters()]


def test_the_critic_receives_no_gradient_from_the_policy_loss():
    """The premise: the two losses reach disjoint parameter sets.

    If they overlapped, the coefficient would balance them and would matter.
    """
    torch.manual_seed(0)
    policy = TreeObsFlowPolicy()
    actor_ids = {id(p) for p in policy.actor.parameters()}
    critic_ids = {id(p) for p in policy.critic.parameters()}
    assert actor_ids and critic_ids
    assert not (actor_ids & critic_ids), (
        "the actor and critic share parameters, so value_loss_coeff would "
        "balance two gradients and this file's premise does not hold"
    )


def test_scaling_the_value_loss_changes_nothing():
    """0.25, 1.0 and 4.0 give the same weights, to the bit."""
    baseline = _run(0.25)
    for coeff in (1.0, 4.0):
        other = _run(coeff)
        assert len(baseline) == len(other)
        differing = [
            i for i, (a, b) in enumerate(zip(baseline, other)) if not torch.equal(a, b)
        ]
        assert not differing, (
            f"value_loss_coeff={coeff} changed {len(differing)} parameters "
            "against 0.25. If this starts failing, either the critic is no "
            "longer disjoint or the optimizer is no longer scale-invariant, "
            "and the docstring above needs rewriting rather than the test."
        )


def test_a_learning_rate_is_not_cancelled():
    """The control, and the thing to reach for instead.

    Adam cancels a constant on the loss. It does not cancel one on the step,
    so a critic that should learn faster or slower needs its own learning
    rate, not a coefficient.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    config = _config()
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
    algo.init_optimizers()
    for group in algo.optimizer.param_groups:
        group["lr"] = group["lr"] * 10
    _collect_and_learn(algo, _tree_env_obs)
    faster = [p.detach().clone() for p in algo.policy.parameters()]

    baseline = _run(0.25)
    assert any(not torch.equal(a, b) for a, b in zip(baseline, faster)), (
        "a ten-fold learning rate changed nothing either, which would mean "
        "the test is not exercising the optimizer at all"
    )
