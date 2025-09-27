from typing import Dict, Type
from loguru import logger
from vlarl_launcher.policy.base_policy import BasePolicy, BasePolicyConfig

class PolicySpec:
    def __init__(self, uid: str, cls: Type[BasePolicy], default_kwargs: dict | None = None):
        self.uid = uid
        self.cls = cls
        self.default_kwargs = default_kwargs or {}
        
    def create(self, **kwargs) -> BasePolicy:
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)
        return self.cls(**_kwargs)

REGISTERED_POLICY_CONFIGS: Dict[str, BasePolicyConfig] = {}
REGISTERED_POLICIES: Dict[str, PolicySpec] = {}

def register_policy(uid: str, override: bool = False, **default_kwargs):
    def _register_policy(cls):
        if uid in REGISTERED_POLICIES and not override:
            raise KeyError(f"Policy {uid} is already registered.")
        if not issubclass(cls, BasePolicy):
            raise TypeError(f"Policy {uid} must inherit from BasePolicy")
        REGISTERED_POLICIES[uid] = PolicySpec(
            uid,
            cls,
            default_kwargs=default_kwargs,
        )
        return cls
    return _register_policy
    
def register_policy_config(uid: str, supported_algos: list[tuple[str, str]] | None = None):
    def _register_policy_config(cls):
        if uid in REGISTERED_POLICY_CONFIGS:
            raise KeyError(f"Policy config {uid} is already registered.")
        REGISTERED_POLICY_CONFIGS[uid] = cls(supported_algos=supported_algos)
        return cls
    return _register_policy_config

def make_policy(uid: str, config: BasePolicyConfig) -> BasePolicy:
    if uid not in REGISTERED_POLICIES:
        raise KeyError(f"Policy {uid} is not registered.")
    policy_cls = REGISTERED_POLICIES[uid]
    return policy_cls.create(config=config)