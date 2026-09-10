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
    """Base class for policies the training server serves.

    **Convention: set `action_dim` and `action_horizon` in `__init__`.**

    Every policy in this package already does - dummy, dppo, fpo, openpi and
    the diffusion base - it was simply never written down. It matters now
    because `plugrl_server.server.metadata` publishes both in the `metadata`
    message the server sends before a client can say anything, and that
    message is the only way a client learns the shape of the actions it is
    about to receive without being told out of band. See SPEC.md section 5.1
    in plugrl-protocol.

    Neither is required. A policy that declares neither still works; its
    clients just have to be configured with the shape by hand, and the
    metadata message omits the keys rather than guessing at them.
    """

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
