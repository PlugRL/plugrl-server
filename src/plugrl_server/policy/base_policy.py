import abc
import dataclasses
import tyro
from typing import Any

from plugrl_server.common.logging_utils import get_logger
from plugrl_server.policy.state import NumpyState, PolicyRuntimeState

logger = get_logger(__name__)


@dataclasses.dataclass
class BasePolicyConfig:
    algo: tyro.conf._markers.Suppress[str] = "unknown"


class BasePolicy(abc.ABC):
    def __init__(self, config: BasePolicyConfig):
        self.config = config
        logger.debug("Initialized policy %s", self.__class__.__name__)

    def prepare_observation(self, _obs: dict[str, Any]) -> NumpyState: ...

    @abc.abstractmethod
    def get_action_and_runtime_state(
        self, _obs: dict[str, Any]
    ) -> tuple[Any, PolicyRuntimeState]: ...

    @abc.abstractmethod
    def fake_runtime_state(self, batch_size: int) -> PolicyRuntimeState: ...

    def get_value(self, _obs: dict[str, Any]) -> Any: ...

    def _get_value(self, obs: Any) -> Any: ...

__all__ = [
    "BasePolicy",
    "BasePolicyConfig",
    "PolicyRuntimeState",
]
