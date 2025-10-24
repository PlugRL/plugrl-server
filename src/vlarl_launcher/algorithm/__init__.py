from .dummy_algorithm import DummyAlgorithm
from .ppo_discrete.ppo_discrete_config import PPODiscreteAlgoConfig

try:
    from .dppo.dppo_config import DPPOAlgoConfig
except ImportError:
    pass