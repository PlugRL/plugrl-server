"""Half-precision weights have to be able to learn from small steps.

Pi0Policy holds its action expert's linear weights in bfloat16. The spacing
between neighbouring bfloat16 values is about 0.8% of the value, so for a
weight of 0.026 it is about 1.2e-4. An Adam step at a learning rate of 1e-5
moves a weight by about 1e-5, which rounds away. With nothing kept in higher
precision, those steps never add up either.

Measured on the real policy, after one E11 harness learn of 32 steps at a
learning rate of 1e-5: 4.7% of the expert's bfloat16 linear parameters had
changed, against 99.1% of its float32 norm parameters.

MasterWeights keeps a float32 copy of every trainable half-precision
parameter. The optimizer steps the copy, and the copy is rounded back into the
model after each step, so small steps accumulate until they cross a bfloat16
value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.master_weights import MasterWeights
from test_fpo_tree_observations import (
    ACTION_DIM,
    HORIZON,
    TreeObsFlowPolicy,
    _collect_and_learn,
    _config,
    _tree_env_obs,
)

LR = 1e-5
WEIGHT = 0.026
STEPS = 40


def _bf16_linear() -> nn.Linear:
    layer = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.fill_(WEIGHT)
    return layer.to(torch.bfloat16)


def _constant_gradient_steps(params, optimizer, step_fn, steps: int = STEPS) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_fn()


class TestTheProblem:
    def test_adam_on_bfloat16_weights_does_not_move_them(self):
        """What the policy did before: every step rounds away."""
        layer = _bf16_linear()
        before = layer.weight.detach().clone()
        optimizer = torch.optim.Adam(layer.parameters(), lr=LR)

        def step():
            layer.weight.grad = torch.ones_like(layer.weight)
            optimizer.step()

        _constant_gradient_steps(layer.parameters(), optimizer, step)

        assert torch.equal(layer.weight, before)


class TestMasterWeights:
    def test_small_steps_accumulate_and_reach_the_model(self):
        layer = _bf16_linear()
        before = layer.weight.detach().clone()
        master = MasterWeights(layer.parameters())
        optimizer = torch.optim.Adam(master.optimizer_params, lr=LR)

        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            master.clear_model_grads()
            layer.weight.grad = torch.ones_like(layer.weight)
            master.grads_to_masters()
            optimizer.step()
            master.masters_to_model()

        (copy,) = master.optimizer_params
        assert copy.dtype == torch.float32
        assert layer.weight.dtype == torch.bfloat16
        moved = float((before.float() - copy.detach()).abs().max())
        assert moved > (STEPS - 1) * LR, f"the float32 copy moved only {moved}"
        assert not torch.equal(layer.weight, before), "the model never saw the steps"
        assert torch.equal(layer.weight, copy.detach().to(torch.bfloat16))

    def test_float32_parameters_are_stepped_directly(self):
        layer = nn.Linear(4, 4)
        master = MasterWeights(layer.parameters())
        assert master.optimizer_params == list(layer.parameters())

    def test_frozen_parameters_are_left_out(self):
        frozen = _bf16_linear()
        frozen.weight.requires_grad_(False)
        trainable = nn.Linear(4, 4)
        master = MasterWeights([*frozen.parameters(), *trainable.parameters()])
        assert master.optimizer_params == list(trainable.parameters())

    def test_grads_are_released_from_the_model_once_copied(self):
        layer = _bf16_linear()
        master = MasterWeights(layer.parameters())
        layer.weight.grad = torch.ones_like(layer.weight)
        master.grads_to_masters()
        assert layer.weight.grad is None
        (copy,) = master.optimizer_params
        assert copy.grad is not None and copy.grad.dtype == torch.float32

    def test_a_parameter_with_no_gradient_leaves_its_copy_without_one(self):
        layer = _bf16_linear()
        master = MasterWeights(layer.parameters())
        master.grads_to_masters()
        (copy,) = master.optimizer_params
        assert copy.grad is None

    def test_sync_from_model_after_loading_weights(self):
        layer = _bf16_linear()
        master = MasterWeights(layer.parameters())
        with torch.no_grad():
            layer.weight.fill_(0.5)
        master.sync_from_model()
        (copy,) = master.optimizer_params
        assert torch.equal(copy.detach(), torch.full_like(copy, 0.5))


class Bf16ActorPolicy(TreeObsFlowPolicy):
    """The tree-observation policy with its actor held in bfloat16, as pi0's expert is."""

    def __init__(self) -> None:
        super().__init__()
        self.actor = self.actor.to(torch.bfloat16)

    def _predict_v(self, x, t, cond, *, cond_cache=None) -> torch.Tensor:
        features = self._features(cond, cond_cache, x.shape[0])
        inputs = torch.cat([features, x.flatten(1), t.reshape(-1, 1)], dim=1)
        return self.actor(inputs.to(torch.bfloat16)).float().reshape(x.shape)


class TestFPOUsesMasterWeights:
    def test_fpo_steps_float32_copies_of_the_bfloat16_actor(self):
        torch.manual_seed(0)
        policy = Bf16ActorPolicy()
        algo = FPOAlgorithm(_config(), policy)
        initial = [p.detach().float().clone() for p in policy.actor.parameters()]

        _collect_and_learn(algo, _tree_env_obs)

        stepped = set(
            id(p) for group in algo.optimizer.param_groups for p in group["params"]
        )
        for model_param, copy in algo.master_weights.pairs:
            assert id(copy) in stepped
            assert id(model_param) not in stepped
            assert model_param.dtype == torch.bfloat16
            assert torch.equal(model_param, copy.detach().to(torch.bfloat16))

        actor_copies = [
            copy
            for model_param, copy in algo.master_weights.pairs
            if any(model_param is p for p in policy.actor.parameters())
        ]
        assert len(actor_copies) == len(initial)
        assert any(
            not torch.equal(copy.detach(), start)
            for copy, start in zip(actor_copies, initial)
        ), "no float32 copy of the actor moved during learn"
        assert policy.actor.weight.shape[0] == HORIZON * ACTION_DIM
        assert np.isfinite(float(policy.actor.weight.float().abs().max()))
