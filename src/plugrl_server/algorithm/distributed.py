import abc

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm
from plugrl_server.common.checkpoint_manager import Checkpoint


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
