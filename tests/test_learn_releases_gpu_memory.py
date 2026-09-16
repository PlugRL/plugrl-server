"""A learn step must hand its memory back before the next inference.

E11's Stage C training died four minutes into its second iteration:

    torch.OutOfMemoryError: Tried to allocate 124.00 MiB. GPU 0 has a total
    capacity of 23.56 GiB of which 26.81 MiB is free.

Nothing had leaked. A learn step allocates far more than inference does -
master weights, optimizer state, the whole buffer's observations, attention
intermediates - and when it ends, PyTorch keeps those blocks in its caching
allocator rather than returning them. Collection then asks for 124 MiB and
there is no block that size left. GPU 0 sat at 23,558 MiB through the learn
and reached 24,098 MiB, of 24,125 MiB, at the moment it died.

The single-iteration harness checks never saw it: a run that stops after its
only learn step never infers again afterwards.

So FPO releases the allocator's cache once a learn step is over. It costs the
next allocations a little time and changes nothing numerically.
"""

from __future__ import annotations

import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm


class _Policy:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)


class _Algorithm(FPOAlgorithm):
    """FPOAlgorithm's post_learn, without building a buffer or an optimizer."""

    def __init__(self, device: str) -> None:  # noqa: D107 - deliberately not super().__init__
        self.policy = _Policy(device)
        self._episode_metric_window = _Window()


class _Window:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_a_cuda_learn_releases_the_allocator_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    algorithm = _Algorithm("cuda")
    algorithm.post_learn()

    assert calls, "the cache was not released after a learn step"


def test_a_cpu_learn_does_not_touch_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    algorithm = _Algorithm("cpu")
    algorithm.post_learn()

    assert not calls, "a CPU policy has no allocator cache to release"


def test_the_episode_window_is_still_reset(monkeypatch):
    """post_learn's existing job, which the override must keep doing."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    algorithm = _Algorithm("cuda")
    algorithm.post_learn()

    assert algorithm._episode_metric_window.reset_calls == 1
