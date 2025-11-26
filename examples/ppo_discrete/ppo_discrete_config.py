import dataclasses
from .ppo_discrete import PPODiscreteAlgoConfig, UID
from vlarl_launcher.algorithm.registration import register_algo_config

@register_algo_config(UID, "atari")
@dataclasses.dataclass
class PPODiscreteAlgoConfigAtari(PPODiscreteAlgoConfig):
    ...
    
@register_algo_config(UID, "classic")
@dataclasses.dataclass
class PPODiscreteAlgoConfigClassic(PPODiscreteAlgoConfig):
    clip_coef: float = 0.1
    buffer_size: int = 512
    
    batch_size: int = 128
    total_steps: int = 500000
    save_interval: int = 50000