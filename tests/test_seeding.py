"""`--seed` used to reach the experiment's name and nothing else.

These pin the two halves of the fix: that seeding actually fixes the draws,
and that the server applies it before it builds anything that draws.
"""

from __future__ import annotations

import inspect
import random

import numpy as np
import torch

from plugrl_server.common.seeding import seed_everything


def _draw() -> tuple[float, float, float]:
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def test_same_seed_gives_the_same_draws():
    seed_everything(1234)
    first = _draw()
    seed_everything(1234)
    second = _draw()
    assert first == second


def test_different_seeds_give_different_draws():
    seed_everything(1)
    first = _draw()
    seed_everything(2)
    second = _draw()
    # All three generators should move; comparing the tuple would pass if only
    # one of them did.
    assert all(a != b for a, b in zip(first, second, strict=True))


def test_seeding_covers_every_generator_the_server_uses():
    """torch, numpy and the stdlib are all drawn from in src/plugrl_server."""
    seed_everything(7)
    torch_first = torch.randn(4).tolist()
    numpy_first = np.random.randint(0, 1000, size=4).tolist()
    python_first = [random.randint(0, 1000) for _ in range(4)]

    seed_everything(7)
    assert torch.randn(4).tolist() == torch_first
    assert np.random.randint(0, 1000, size=4).tolist() == numpy_first
    assert [random.randint(0, 1000) for _ in range(4)] == python_first


def test_server_seeds_before_it_builds_the_policy():
    """A policy's own construction draws, so seeding after it would be too late."""
    from plugrl_server import cli

    source = inspect.getsource(cli._main)
    assert "seed_everything(args.seed)" in source, (
        "the server no longer seeds; --seed would again only reach the run's name"
    )
    assert source.index("seed_everything(args.seed)") < source.index("make_policy("), (
        "seeding must happen before the policy is constructed"
    )
