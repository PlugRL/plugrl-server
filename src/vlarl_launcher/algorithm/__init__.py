from .dummy_algorithm import DummyAlgorithm

try:
    from .dppo.dppo_config import DPPOAlgoConfig
except ImportError:
    pass