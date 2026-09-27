"""Freezing part of a policy by parameter name.

E22 located the damage FPO does to pi0.5 in its update to the action
expert's MLP. The test that follows from it is FPO with those matrices
frozen, so the policy needs a way to freeze parameters by name - and a
frozen parameter has to stay frozen through the optimizer path FPO actually
uses, `MasterWeights` and Adam, not only carry `requires_grad = False`.

A pattern that matches nothing raises: a typo would otherwise run the whole
experiment with nothing frozen, and nothing in its results would say so.
"""

from __future__ import annotations

import pytest
import torch

from plugrl_server.algorithm.master_weights import MasterWeights
from plugrl_server.policy.freezing import freeze_matching


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))


class _Expert(torch.nn.Module):
    """Named like pi0's gemma_expert: model.layers.N.{self_attn,mlp}."""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Block() for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            x = block.mlp(block.self_attn(x))
        return x


def test_it_freezes_what_matches_and_nothing_else():
    expert = _Expert()

    frozen = freeze_matching(expert, [r"\.mlp\."])

    names = dict(expert.named_parameters())
    assert sorted(frozen) == sorted(n for n in names if ".mlp." in n)
    assert len(frozen) == 8  # 2 layers x 2 linears x (weight, bias)
    for name, param in names.items():
        assert param.requires_grad == (".mlp." not in name), name


def test_a_pattern_that_matches_nothing_raises():
    expert = _Expert()

    with pytest.raises(ValueError, match="mpl"):
        freeze_matching(expert, [r"\.mlp\.", r"\.mpl\."])
    # Nothing is frozen when the call is refused.
    assert all(p.requires_grad for p in expert.parameters())


def test_no_patterns_freezes_nothing():
    expert = _Expert()

    assert freeze_matching(expert, []) == []
    assert all(p.requires_grad for p in expert.parameters())


def test_frozen_parameters_do_not_move_through_fpo_s_optimizer_path():
    """MasterWeights and Adam, as FPOAlgorithm builds them, on a bf16 model."""
    torch.manual_seed(0)
    expert = _Expert().to(torch.bfloat16)
    freeze_matching(expert, [r"layers\.0\.mlp\."])
    before = {n: p.detach().clone() for n, p in expert.named_parameters()}

    master = MasterWeights(list(expert.parameters()))
    optimizer = torch.optim.Adam(master.optimizer_params, lr=1e-2)
    for _ in range(3):
        loss = expert(torch.randn(16, 4, dtype=torch.bfloat16)).float().pow(2).mean()
        loss.backward()
        master.grads_to_masters()
        optimizer.step()
        optimizer.zero_grad()
        master.masters_to_model()
        master.clear_model_grads()

    for name, param in expert.named_parameters():
        moved = not torch.equal(param.detach(), before[name])
        assert moved == ("layers.0.mlp." not in name), name
