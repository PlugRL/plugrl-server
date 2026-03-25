import abc
import dataclasses
import numpy as np

from plugrl_server.policy.base_policy import BasePolicy
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

    def derive_train_state(self, runtime_state: PolicyRuntimeState) -> PolicyTrainState:
        return None

    def example_train_state(self, batch_size: int) -> PolicyTrainState:
        return None

    def build_step_state_from_runtime_state(
        self,
        runtime_state: PolicyRuntimeState,
        *,
        include_train_state: bool = True,
    ) -> PolicyStepState:
        return PolicyStepState(
            runtime_state=runtime_state,
            train_state=(
                self.derive_train_state(runtime_state) if include_train_state else None
            ),
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
