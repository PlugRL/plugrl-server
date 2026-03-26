import abc
import dataclasses
import numpy as np

from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import MetricDict, merge_metric_groups
from plugrl_server.policy.base_policy import BasePolicy
from plugrl_server.policy.state import (
    PolicyRuntimeState,
    PolicyStepState,
    PolicyTrainState,
    TrainStateSpec,
    infer_train_state_spec,
)
from plugrl_server.common.checkpoint_manager import Checkpoint

logger = get_logger(__name__)


@dataclasses.dataclass
class BaseAlgoConfig:
    global_steps: int | None = None


class EpisodeMetricWindow:
    def __init__(self) -> None:
        self._episodes: list[tuple[float, float, float]] = []

    def record(self, episode_info: dict) -> None:
        self._episodes.append(
            (
                float(episode_info.get("s", 0.0)),
                float(episode_info.get("r", 0.0)),
                float(episode_info.get("l", 0.0)),
            )
        )

    def summary(self) -> dict[str, float]:
        if not self._episodes:
            return dict(success=0.0, reward=0.0, length=0.0)
        stats = np.asarray(self._episodes, dtype=np.float32)
        return dict(
            success=float(stats[:, 0].mean()),
            reward=float(stats[:, 1].mean()),
            length=float(stats[:, 2].mean()),
        )

    def reset(self) -> None:
        self._episodes.clear()


class BaseAlgorithm(abc.ABC):
    break_action_chunk: bool = False

    def __init__(self, config: BaseAlgoConfig, policy: BasePolicy):
        self.config = config
        self.policy = policy
        self._episode_metric_window = EpisodeMetricWindow()
        logger.debug(
            "Initialized algorithm %s with policy %s",
            self.__class__.__name__,
            policy.__class__.__name__,
        )

    def init_optimizers(self) -> None:
        logger.debug("init_optimizers called for %s", self.__class__.__name__)

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        return None

    def example_train_state(self, batch_size: int) -> PolicyTrainState:
        return None

    def example_train_state_spec(self) -> TrainStateSpec | None:
        logger.debug("Inferring train_state_spec for %s", self.__class__.__name__)
        return infer_train_state_spec(self.example_train_state(batch_size=1))

    def build_step_state_from_runtime_state(
        self,
        runtime_state: PolicyRuntimeState,
        *,
        include_train_state: bool = True,
    ) -> PolicyStepState:
        logger.debug(
            "Building step state for %s include_train_state=%s",
            self.__class__.__name__,
            include_train_state,
        )
        return PolicyStepState(
            runtime_state=runtime_state,
            train_state=(
                self.derive_train_state(runtime_state) if include_train_state else None
            ),
        )

    def record_episode_metrics(self, episode_info: dict) -> None:
        self._episode_metric_window.record(episode_info)

    def get_rollout_metrics(self, prefix: str = "rollout") -> MetricDict:
        summary = self._episode_metric_window.summary()
        return dict(
            **{
                prefix: dict(
                    success=summary["success"],
                    reward=summary["reward"],
                    length=summary["length"],
                )
            }
        )

    def reset_episode_metrics(self) -> None:
        self._episode_metric_window.reset()

    def get_lifecycle_metrics(self) -> MetricDict:
        return dict()

    def get_total_training_steps(self) -> int | None:
        return self.config.global_steps

    def build_train_info(self, *metric_groups: MetricDict) -> MetricDict:
        return merge_metric_groups(
            *metric_groups,
            self.get_lifecycle_metrics(),
            self.get_rollout_metrics(),
        )

    @abc.abstractmethod
    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]: ...

    @abc.abstractmethod
    def feedback(
        self,
        *,
        obs: dict,
        runtime_state: PolicyRuntimeState,
        train_state: PolicyTrainState = None,
        terminated: bool,
        truncated: bool,
        next_obs: dict,
        reward: float,
        info: dict,
        next_terminated: bool,
        next_truncated: bool,
        prev_node: tuple,
    ) -> tuple[tuple, int, dict]: ...

    def learn(self) -> tuple[int, MetricDict]:
        step, algorithm_metrics = self.learn_impl()
        return step, self.build_train_info(algorithm_metrics)

    @abc.abstractmethod
    def learn_impl(self) -> tuple[int, MetricDict]: ...

    @abc.abstractmethod
    def should_learn(self) -> bool: ...

    @abc.abstractmethod
    def should_stop(self) -> bool: ...

    @abc.abstractmethod
    def should_save(self) -> bool: ...

    @abc.abstractmethod
    def create_checkpoint(self) -> Checkpoint: ...

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint: Checkpoint) -> None: ...

    def pre_learn(self) -> None:
        logger.debug("pre_learn called for %s", self.__class__.__name__)

    def post_learn(self) -> None:
        logger.debug("post_learn called for %s", self.__class__.__name__)
