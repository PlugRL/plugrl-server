import dataclasses
from typing import Type, Dict
from loguru import logger
from vlarl_launcher.policy.base_policy import BasePolicy, BasePolicyConfig

from vlarl_launcher.algorithm.base import BaseAlgorithm, BaseAlgoConfig
class AlgoSpec:
    def __init__(self, uid: str, cls: Type[BaseAlgorithm], default_kwargs: dict | None = None):
        self.uid = uid
        self.cls = cls
        self.default_kwargs = default_kwargs or {}
        
    def create(self, **kwargs) -> BaseAlgorithm:
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)
        return self.cls(**_kwargs)

REGISTERED_ALGO_CONFIGS: Dict[str, BaseAlgoConfig] = {}
REGISTERED_ALGORITHMS: Dict[str, AlgoSpec] = {}

def register_algo(uid: str, override: bool = False, **default_kwargs):
    def _register_algo(cls):
        if uid in REGISTERED_ALGORITHMS and not override:
            raise KeyError(f"Algorithm {uid} is already registered.")
        if not issubclass(cls, BaseAlgorithm):
            raise TypeError(f"Algorithm {uid} must inherit from BaseAlgorithm")
        REGISTERED_ALGORITHMS[uid] = AlgoSpec(
            uid,
            cls,
            default_kwargs=default_kwargs,
        )
        return cls
    return _register_algo

def make_algo(uid: str, policy: BasePolicy, **kwargs) -> BaseAlgorithm:
    if uid not in REGISTERED_ALGORITHMS:
        raise KeyError(f"Algorithm {uid} is not registered.")
    return REGISTERED_ALGORITHMS[uid].create(policy=policy, **kwargs)

def register_algo_config(uid: str):
    def _register_algo_config(cls):
        if uid in REGISTERED_ALGO_CONFIGS:
            raise KeyError(f"Algo config {uid} is already registered.")
        REGISTERED_ALGO_CONFIGS[uid] = cls()
        return cls
    return _register_algo_config