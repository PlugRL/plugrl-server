import abc
import dataclasses
from ..registration import register_policy, register_policy_config
from ..base_policy import BasePolicyConfig, BasePolicy, InternalState

UID = "atari-policy"

@register_policy_config(UID, supported_algos=["ppo-discrete"])
@dataclasses.dataclass
class AtariPolicyConfig(BasePolicyConfig):
    ...

@register_policy(UID)
class AtariPolicy(BasePolicy):
    def __init__(self, config: AtariPolicyConfig):
        self.config = config

    def infer(self, obs: dict) -> dict:
        ...

    def get_value(self, obs: dict):
        ...
        
    def get_action_and_value(self, obs: dict, internal_state: InternalState) -> tuple:
        ...