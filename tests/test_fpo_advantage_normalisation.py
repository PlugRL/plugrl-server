"""Advantages are normalised once, over the buffer, not once per minibatch.

Per-minibatch normalisation is the usual arrangement and it is fine at the
batch sizes FPO's own defaults assume - 1024. A pi0.5-sized policy does not
fit those. E14 ran at batch 8, forced by memory, and dividing by the standard
deviation of eight samples is not a rescaling; it is a source of noise, and
subtracting their mean forces half of every eight positive and half negative
at unit scale whatever the rewards were.

These tests pin where the division happens and that both paths through
`learn_impl` take it, not how well it works.
"""

from __future__ import annotations

import inspect

import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from test_fpo_tree_observations import TreeObsFlowPolicy, _config

NORMALISE = "(advantage - advantage.mean())"


def _algo(**overrides) -> FPOAlgorithm:
    config = _config()
    for key, value in overrides.items():
        setattr(config, key, value)
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
    algo.init_optimizers()
    return algo


def test_minibatch_loss_does_not_normalise_the_advantage():
    """`_compute_loss` sees one minibatch and must not rescale it.

    Source inspection, because reaching the arithmetic any other way means
    running a policy forward. Same approach as tests/test_seeding.py. It has
    to match the normalising expression rather than `advantage.std()`, which
    also appears there legitimately as a logged metric.
    """
    source = inspect.getsource(FPOAlgorithm._compute_loss)
    assert NORMALISE not in source, (
        "_compute_loss normalises the advantage again. It receives one "
        "minibatch, and at the batch sizes a VLA forces that manufactures "
        "signal rather than removing scale."
    )


def test_both_paths_through_learn_take_the_same_scaling():
    """`fpo_playground_trick` selects between two sources of advantages.

    With it on, they come from `_refresh_epoch_value_targets` once per epoch;
    with it off, straight out of the buffer. Moving the normalising out of
    `_compute_loss` silently leaves the second path unnormalised unless it
    calls the helper too.
    """
    source = inspect.getsource(FPOAlgorithm.learn_impl)
    assert source.count("_scale_advantage") >= 1, "the off path lost its normalising"
    refresh = inspect.getsource(FPOAlgorithm._refresh_epoch_value_targets)
    assert "_scale_advantage" in refresh


def test_the_raw_spread_is_recorded_before_the_division():
    """The statistic that shows whether the buffer held a signal at all.

    After normalising it is 1.0 by construction. E14 logged only that one, so
    nine iterations of metrics could not show that from the third iteration on
    there was no reward anywhere in the buffer.
    """
    source = inspect.getsource(FPOAlgorithm._scale_advantage)
    assert source.index("_advantage_raw_std") < source.index(NORMALISE)


def test_scaling_centres_and_rescales_the_whole_tensor():
    algo = _algo()
    advantage = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])
    scaled = algo._scale_advantage(advantage)
    assert abs(float(scaled.mean())) < 1e-5
    assert abs(float(scaled.std()) - 1.0) < 1e-3


def test_a_minibatch_of_the_result_is_not_itself_unit_scale():
    """The point of the change: a minibatch keeps its deviation from the buffer.

    Under per-minibatch normalising every slice came out centred with unit
    spread, which is the behaviour that manufactures signal at batch 8.
    """
    algo = _algo()
    advantage = torch.arange(64, dtype=torch.float32)
    scaled = algo._scale_advantage(advantage)
    first_eight = scaled[:8]
    assert abs(float(first_eight.mean())) > 0.5, "the slice was re-centred"
    assert abs(float(first_eight.std()) - 1.0) > 0.1, "the slice was re-scaled"


def test_the_flag_still_turns_it_off():
    algo = _algo(normalize_advantage=False)
    advantage = torch.tensor([5.0, 7.0, 9.0, 11.0])
    scaled = algo._scale_advantage(advantage)
    assert torch.equal(scaled, advantage)


def test_the_raw_spread_is_recorded_even_when_scaling_is_off():
    algo = _algo(normalize_advantage=False)
    algo._scale_advantage(torch.tensor([0.0, 2.0, 4.0, 6.0]))
    assert abs(algo._advantage_raw_std - 2.5819888) < 1e-4


def test_a_buffer_with_no_spread_is_reported_as_such():
    """What E14 needed and did not have.

    A buffer where every advantage is identical carries no signal. The raw
    spread says so; the normalised one cannot, and this is the number that
    would have made nine iterations of zero reward visible in the metrics.
    """
    algo = _algo()
    algo._scale_advantage(torch.full((32,), 3.0))
    assert algo._advantage_raw_std == 0.0


def test_the_default_config_still_normalises():
    assert FPOAlgoConfig.normalize_advantage is True
