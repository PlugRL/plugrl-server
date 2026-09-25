"""Freeze a module's parameters by name.

A frozen parameter gets `requires_grad = False`. That is enough for every
training algorithm here: `MasterWeights` leaves such parameters out of the
optimizer entirely, so they receive no update, no weight decay and no
optimizer state.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


def freeze_matching(module: torch.nn.Module, patterns: Sequence[str]) -> list[str]:
    """Stop gradients to every parameter whose name matches any of `patterns`.

    Names are the module's own `named_parameters()`; each pattern is a regular
    expression searched anywhere in the name. Returns the names frozen.

    Raises if any pattern matches no parameter, before freezing anything: a
    typo would otherwise leave the model fully trainable, and a run meant to
    test a frozen part would test nothing.
    """
    compiled = [(p, re.compile(p)) for p in patterns]
    named = list(module.named_parameters())
    unmatched = [p for p, rx in compiled if not any(rx.search(n) for n, _ in named)]
    if unmatched:
        raise ValueError(f"these patterns match no parameter: {unmatched}")

    frozen = []
    for name, param in named:
        if any(rx.search(name) for _, rx in compiled):
            param.requires_grad = False
            frozen.append(name)
    return frozen
