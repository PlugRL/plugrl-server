import dataclasses
from typing import Any, Literal

import torch
import torch.nn as nn

from plugrl_server.common.data_utils import numpy_state_to_torch_tree
from plugrl_server.common.logging_utils import get_logger
from .base_policy import BasePolicy, BasePolicyConfig

logger = get_logger(__name__)


@dataclasses.dataclass
class BaseTorchPolicyConfig(BasePolicyConfig):
    device: torch.device | Literal["cpu", "cuda"] = "cuda"


class BaseTorchPolicy(BasePolicy, nn.Module):
    def __init__(self, config: BaseTorchPolicyConfig):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, config)
        self.device = (
            config.device
            if isinstance(config.device, torch.device)
            else torch.device(config.device)
        )
        logger.debug(
            "Initialized torch policy %s on device=%s",
            self.__class__.__name__,
            self.device,
        )

    def extract_model_obs_tensor(self, _obs: dict[str, Any]) -> Any:
        return numpy_state_to_torch_tree(self.prepare_observation(_obs))


__all__ = [
    "BaseTorchPolicy",
    "BaseTorchPolicyConfig",
]
