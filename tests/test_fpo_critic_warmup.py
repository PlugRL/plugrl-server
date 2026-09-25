"""During a critic warmup the value head learns and the policy does not.

FPO weights every policy update by an advantage, and an advantage is a return
minus the value head's estimate of it. The head starts random, so on the first
iteration those weights are noise. E14 measured `losses/value_loss` at 0.880
on its first iteration and 9.3e-05 by its ninth, and the policy went from 29
of 50 to 0 of 50 across the first update.

These tests pin the mechanism, not whether it helps: the actor must come out
of a warmup iteration exactly as it went in, and the critic must not.
"""

from __future__ import annotations

import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from test_fpo_tree_observations import (
    TreeObsFlowPolicy,
    _collect_and_learn,
    _config,
    _tree_env_obs,
)


def _algo(**overrides) -> FPOAlgorithm:
    config = _config()
    for key, value in overrides.items():
        setattr(config, key, value)
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
    algo.init_optimizers()
    return algo


def _clone(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], module) -> list[str]:
    after = module.state_dict()
    return [k for k, v in before.items() if not torch.equal(v, after[k])]


def test_the_default_is_no_warmup():
    assert FPOAlgoConfig.n_critic_warmup_itrs == 0
    assert not _algo().in_critic_warmup()


def test_warmup_covers_the_first_iterations_and_then_stops():
    algo = _algo(n_critic_warmup_itrs=2)
    assert algo.curr_train_itrs == 0
    assert algo.in_critic_warmup()
    algo.curr_train_itrs = 1
    assert algo.in_critic_warmup()
    algo.curr_train_itrs = 2
    assert not algo.in_critic_warmup(), "warmup ran one iteration too long"


def test_zeroing_leaves_the_critic_gradients_alone():
    algo = _algo(n_critic_warmup_itrs=1)
    for param in algo.policy.actor.parameters():
        param.grad = torch.ones_like(param)
    for param in algo.policy.critic.parameters():
        param.grad = torch.ones_like(param)

    algo._zero_actor_grads()

    assert all(float(p.grad.abs().sum()) == 0.0 for p in algo.policy.actor.parameters())
    assert all(
        float(p.grad.abs().sum()) > 0.0 for p in algo.policy.critic.parameters()
    ), "the warmup zeroed the head it is supposed to be training"


def test_a_warmup_iteration_moves_the_critic_and_not_the_actor():
    """The whole point, end to end through a real learn step."""
    algo = _algo(n_critic_warmup_itrs=1)
    actor_before = _clone(algo.policy.actor)
    critic_before = _clone(algo.policy.critic)

    _collect_and_learn(algo, _tree_env_obs)

    actor_moved = _changed(actor_before, algo.policy.actor)
    critic_moved = _changed(critic_before, algo.policy.critic)
    assert actor_moved == [], f"the actor moved during a warmup: {actor_moved[:4]}"
    assert critic_moved, "the critic did not learn during its own warmup"


def test_without_warmup_the_actor_does_move():
    """The control: the same learn step, warmup off, must move the actor.

    Without this the test above passes just as well on a policy that never
    learns anything.
    """
    algo = _algo(n_critic_warmup_itrs=0)
    actor_before = _clone(algo.policy.actor)

    _collect_and_learn(algo, _tree_env_obs)

    assert _changed(actor_before, algo.policy.actor), (
        "the actor did not move with the warmup off, so the test above proves nothing"
    )
