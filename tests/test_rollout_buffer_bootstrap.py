"""Bootstrap values have to land in the buffer's own value layout.

`GAEBuffer.compute_advantages_and_returns` asks the policy for the value of
every observation that ends a chain without a successor, and writes the
answers into `last_values`. It reshaped them to `(-1, 1)` first, which is
FPO's layout - FPO stores values as `(N, 1)`. DPPO stores them as `(N,)`.

With one bootstrap value per chunk numpy drops the leading unit dimension and
the assignment happens to succeed, so DPPO ran a hundred iterations in E17
without meeting it. With two in one chunk it raises. Found while testing
DPPO's observation statistics with two envs whose episodes end together.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.buffer.rollout_buffer import GAEBuffer

OBS = 3
BUFFER = 8


class _KnownValues:
    """A stand-in whose value for an observation is its first coordinate.

    Observations arrive as the env client sends them - a mapping - and are
    concatenated per key, so this reads the key it was given.
    """

    def get_value(self, obs: dict) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs["v"])[:, :1], dtype=torch.float32)


def _train_state(value_shape: tuple[int, ...], k: int) -> dict:
    return dict(
        obs=np.full((1, OBS), float(k), dtype=np.float32),
        action=np.zeros((1, 2), dtype=np.float32),
        logprob=np.zeros((1, 2), dtype=np.float32),
        value=np.zeros((1,) + value_shape, dtype=np.float32),
    )


def _buffer_of_episode_ends(value_shape: tuple[int, ...], ends: int) -> GAEBuffer:
    """`ends` one-frame episodes, each needing a bootstrap value."""
    buffer = GAEBuffer(BUFFER, _train_state(value_shape, 0))
    for k in range(ends):
        node = buffer.add_frame(
            prev_node=(-1, buffer.buffer_signature),
            train_state=_train_state(value_shape, k),
            reward=0.0,
            terminated=False,
            truncated=False,
            last_value=None,
            next_terminated=False,
            next_truncated=False,
        )
        # The observation after frame k is worth 10 + k.
        buffer.add_next_obs_value_request(
            obs={"v": np.full((1, OBS), 10.0 + k, dtype=np.float32)}, end_node=node
        )
    return buffer


@pytest.mark.parametrize(
    "value_shape, algorithm",
    [((1,), "FPO stores (N, 1)"), ((), "DPPO stores (N,)")],
)
@pytest.mark.parametrize("ends", [1, 2, 5])
def test_every_bootstrap_value_lands_where_it_belongs(value_shape, algorithm, ends):
    buffer = _buffer_of_episode_ends(value_shape, ends)

    buffer.compute_advantages_and_returns(policy=_KnownValues(), batch_size=4)

    got = buffer.last_values[:ends].reshape(-1)
    expected = 10.0 + np.arange(ends, dtype=np.float32)
    np.testing.assert_allclose(got, expected, err_msg=algorithm)


def test_two_in_one_chunk_is_the_case_that_used_to_raise():
    """Named on its own because it is the regression, not a parameter of one.

    One value per chunk passed before too; this is the shape that did not.
    """
    buffer = _buffer_of_episode_ends((), 2)

    buffer.compute_advantages_and_returns(policy=_KnownValues(), batch_size=4)

    np.testing.assert_allclose(buffer.last_values[:2], [10.0, 11.0])
