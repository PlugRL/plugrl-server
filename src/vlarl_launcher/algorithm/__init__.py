from .dummy_algorithm import DummyAlgorithm
from .ppo_discrete.ppo_discrete_config import PPODiscreteAlgoConfig

try:
    from .dppo.dppo import DPPOAlgoConfig
except ImportError:
    pass

try:
    from .grpo_diffusion.grpo_diffusion import GRPODiffusionAlgoConfig
except ImportError:
    pass