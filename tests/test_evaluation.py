import numpy as np
import pytest

from plugrl_server.algorithm.evaluation import EvalConfig, Evaluation
from plugrl_server.policy.base_policy import BasePolicy, BasePolicyConfig


class _StubPolicy(BasePolicy):
    def get_action_and_runtime_state(self, _obs: dict) -> tuple[np.ndarray, None]:
        return np.zeros((1, 1), dtype=np.float32), None

    def fake_runtime_state(self, _batch_size: int) -> None:
        return None


def _make_algo(*, num_episodes: int | None = None) -> Evaluation:
    return Evaluation(
        config=EvalConfig(num_episodes=num_episodes),
        policy=_StubPolicy(BasePolicyConfig()),
    )


def test_eval_runs_indefinitely_by_default() -> None:
    algo = _make_algo()

    assert algo.get_collect_progress_total() is None
    assert algo.get_collect_progress_completed() == 0
    assert not algo.should_stop()

    algo.feedback(
        obs=dict(),
        runtime_state=None,
        terminated=False,
        truncated=False,
        next_obs=dict(),
        reward=1.0,
        info=dict(episode=dict(r=1.0, l=5, s=1.0)),
        next_terminated=True,
        next_truncated=False,
        prev_node=(-1, ""),
    )

    assert algo.get_collect_progress_completed() == 1
    assert not algo.should_stop()


def test_eval_stops_after_configured_episode_count() -> None:
    algo = _make_algo(num_episodes=2)

    algo.feedback(
        obs=dict(),
        runtime_state=None,
        terminated=False,
        truncated=False,
        next_obs=dict(),
        reward=0.0,
        info=dict(),
        next_terminated=False,
        next_truncated=False,
        prev_node=(-1, ""),
    )
    assert algo.get_collect_progress_completed() == 0
    assert not algo.should_stop()

    algo.feedback(
        obs=dict(),
        runtime_state=None,
        terminated=False,
        truncated=False,
        next_obs=dict(),
        reward=1.0,
        info=dict(),
        next_terminated=True,
        next_truncated=False,
        prev_node=(-1, ""),
    )
    assert algo.get_collect_progress_completed() == 1
    assert not algo.should_stop()

    algo.feedback(
        obs=dict(),
        runtime_state=None,
        terminated=False,
        truncated=False,
        next_obs=dict(),
        reward=1.0,
        info=dict(episode=dict(r=2.0, l=7, s=0.0, mask=True)),
        next_terminated=False,
        next_truncated=True,
        prev_node=(-1, ""),
    )
    assert algo.get_collect_progress_total() == 2
    assert algo.get_collect_progress_completed() == 2
    assert algo.should_stop()


def test_eval_config_rejects_negative_num_episodes() -> None:
    with pytest.raises(ValueError, match="num_episodes"):
        EvalConfig(num_episodes=-1)
