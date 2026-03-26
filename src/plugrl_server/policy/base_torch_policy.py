import dataclasses
from typing import Literal

import torch
import torch.nn as nn

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


__all__ = [
    "BaseTorchPolicy",
    "BaseTorchPolicyConfig",
]
