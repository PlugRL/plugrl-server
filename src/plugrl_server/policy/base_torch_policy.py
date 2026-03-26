import dataclasses
from typing import Literal

import torch
import torch.nn as nn

from .base_policy import BasePolicy, BasePolicyConfig


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


__all__ = [
    "BaseTorchPolicy",
    "BaseTorchPolicyConfig",
]
