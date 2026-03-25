import abc
import dataclasses
import numpy as np

from plugrl_server.policy.base_policy import BasePolicy, InternalState
from plugrl_server.policy.state import PolicyRuntimeState, PolicyStepState, PolicyTrainState
from plugrl_server.common.checkpoint_manager import Checkpoint


@dataclasses.dataclass
class BaseAlgoConfig: ...


class BaseAlgorithm(abc.ABC):
    break_action_chunk: bool = False

    def __init__(self, config: BaseAlgoConfig, policy: BasePolicy):
        self.config = config
        self.policy = policy

    def init_optimizers(self) -> None: ...

    @property
    def active_policy(self) -> BasePolicy:
        return self.policy

    def export_policy_step_state(
        self,
        internal_state: InternalState,
        *,
        include_train_state: bool = True,
    ) -> PolicyStepState:
        return self.active_policy.build_policy_step_state(
            internal_state,
            include_train_state=include_train_state,
        )

    def infer_step(self, obs: dict, *, include_train_state: bool = False) -> tuple[np.ndarray, PolicyStepState]:
        action, internal_state = self.infer(obs)
        return action, self.export_policy_step_state(
            internal_state,
            include_train_state=include_train_state,
        )

    @abc.abstractmethod
    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]: ...

    @abc.abstractmethod
    def feedback(
        self,
        *,
        obs: dict,
        internal_state: PolicyRuntimeState,
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

    @abc.abstractmethod
    def learn(self) -> tuple[int, dict]: ...

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

    def pre_learn(self) -> None: ...

    def post_learn(self) -> None: ...


class DDPAlgorithm(BaseAlgorithm):
    ddp_enabled: bool

    @abc.abstractmethod
    def activate_ddp(self, ddp_policy) -> None: ...

    @abc.abstractmethod
    def get_server_data(self) -> tuple[int, dict, dict]: ...

    @abc.abstractmethod
    def load_server_data(
        self, global_step: int, meta_info: dict, data: dict
    ) -> None: ...

    @abc.abstractmethod
    def load_learner_state(self, checkpoint: Checkpoint) -> None: ...

    @abc.abstractmethod
    def create_ddp_checkpoint(self) -> Checkpoint: ...
