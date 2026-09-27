"""GAE has to end an episode at the step that ended it, not one step later.

Both servers (`websocket_agent_server`, `ray_agent_server`) call `feedback`
for step t with

    terminated, truncated            = step t-1's outcome
    next_terminated, next_truncated  = step t's outcome
    prev_node                        = the node returned for step t-1

and never reset `prev_node` at an episode boundary, so the first frame of an
episode is linked to the last frame of the one before it. A frame carrying
`terminated` therefore *begins* an episode; whether its own transition ended
one is in `next_*`.

`GAEBuffer` read the wrong pair for linked frames. Until 48042e5 it cut the
recursion with the successor's `dones[next_idx]`, which under this
convention is step t's own outcome. 48042e5 changed that to the frame's own
`dones[step]` - step t-1's - so since then every episode has been cut one
step late: the step that fell over in Hopper bootstrapped from the value of
the next episode's first state, and each episode's first step was cut off
from everything after it. FPO, DPPO and anything else on `GAEBuffer` all
computed advantages this way.

These feed the buffer exactly as the servers do and check the advantages
against ones worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.buffer.rollout_buffer import GAEBuffer


def _train_state(value: float = 0.0) -> dict:
    return dict(
        obs=np.zeros((1, 2), np.float32),
        action=np.zeros((1, 1), np.float32),
        logprob=np.zeros((1,), np.float32),
        value=np.full((1,), value, np.float32),
    )


class _ConstantValue:
    def __init__(self, value: float) -> None:
        self.value = value

    def get_value(self, obs: dict) -> torch.Tensor:
        return torch.full((np.asarray(obs["v"]).shape[0],), self.value)


def _as_the_server_feeds_it(
    outcomes: list[str],
    *,
    treat_truncated_as_done: bool,
    bootstrap: float = 0.0,
) -> GAEBuffer:
    """One env, reward 1 at every step, every stored value 0, gamma = lambda = 1.

    `outcomes[t]` is how step t ended: "" (it did not), "terminated" or
    "truncated". The client resets after either, and the server keeps the
    chain going across the reset.
    """
    buffer = GAEBuffer(
        len(outcomes),
        _train_state(),
        gamma=1.0,
        gae_lambda=1.0,
        treat_truncated_as_done=treat_truncated_as_done,
    )
    prev_node: tuple = (-1, "")
    previous = ""
    for outcome in outcomes:
        prev_node = buffer.add_frame(
            prev_node=prev_node,
            train_state=_train_state(),
            reward=1.0,
            terminated=previous == "terminated",
            truncated=previous == "truncated",
            last_value=None,
            next_terminated=outcome == "terminated",
            next_truncated=outcome == "truncated",
        )
        buffer.add_next_obs_value_request(
            obs={"v": np.zeros((1, 2), np.float32)}, end_node=prev_node
        )
        previous = outcome
    buffer.compute_advantages_and_returns(
        policy=_ConstantValue(bootstrap), batch_size=4
    )
    return buffer


THREE_EPISODES = ["", "", "terminated"] * 3


@pytest.mark.parametrize("treat_truncated_as_done", [True, False])
def test_each_terminated_episode_is_its_own(treat_truncated_as_done):
    """Three steps of reward 1 and a fall: 3, 2, 1 in every episode.

    Before the fix: 4, 3, 2, 1, 3, 2, 1, 2, 1 - the first fall carried on
    into the second episode, and the last episode lost its first step.
    """
    buffer = _as_the_server_feeds_it(
        THREE_EPISODES, treat_truncated_as_done=treat_truncated_as_done
    )

    np.testing.assert_array_equal(buffer.advantages[:9], [3, 2, 1] * 3)


def test_a_truncated_episode_ends_like_a_terminated_one_when_told_to():
    buffer = _as_the_server_feeds_it(
        ["", "", "truncated"] * 3, treat_truncated_as_done=True
    )

    np.testing.assert_array_equal(buffer.advantages[:9], [3, 2, 1] * 3)


def test_otherwise_the_truncated_step_is_left_out_and_nothing_crosses_it():
    """With treat_truncated_as_done off, the step that was truncated has its
    delta masked to zero - GAEBuffer's treatment - and the episode before it
    still does not reach into the next."""
    buffer = _as_the_server_feeds_it(
        ["", "", "truncated"] * 3, treat_truncated_as_done=False
    )

    np.testing.assert_array_equal(buffer.advantages[:9], [2, 1, 0] * 3)


def test_a_chain_that_runs_off_the_buffer_bootstraps_even_if_it_just_began():
    """The buffer fills on the first step of a new episode. That step did not
    end anything, so it bootstraps from the value of what came after it.

    Before the fix a frame carrying `terminated` at the end of a chain was
    taken as terminal itself, and its bootstrap value was dropped.
    """
    buffer = _as_the_server_feeds_it(
        ["", "", "terminated", ""], treat_truncated_as_done=True, bootstrap=5.0
    )

    np.testing.assert_array_equal(buffer.advantages[:4], [3, 2, 1, 6])


def test_a_chain_that_runs_off_the_buffer_mid_episode_bootstraps():
    buffer = _as_the_server_feeds_it(
        ["", "", "terminated", "", ""], treat_truncated_as_done=False, bootstrap=5.0
    )

    np.testing.assert_array_equal(buffer.advantages[:5], [3, 2, 1, 7, 6])
