import dataclasses

from plugrl_server.algorithm.base_algorithm import BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo_config

UID = "ppo"


@register_algo_config(UID)
@dataclasses.dataclass
class PPOAlgoConfig(BaseAlgoConfig):
    """CleanRL `ppo_continuous_action.py`'s defaults, which it runs on every
    MuJoCo task unchanged.

    CleanRL's rollout is `num_envs * num_steps` = 1 x 2048 transitions, split
    into `num_minibatches` = 32; here that is `buffer_size` 2048 and
    `batch_size` 64, however many environments the client runs. Its
    1,000,000 steps are `train_itrs` 488 iterations of 2048.
    """

    learning_rate: float = 3e-4
    # Linearly to zero across `train_itrs`, as CleanRL's `anneal_lr`.
    anneal_lr: bool = True
    buffer_size: int = 2048
    batch_size: int = 64
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Stop an iteration's epochs once the approximate KL passes this. None,
    # CleanRL's default, never stops.
    target_kl: float | None = None
    # gymnasium's NormalizeReward and TransformReward, which CleanRL wraps its
    # environments in: rewards divided by the running deviation of the
    # discounted return, then clipped.
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    train_itrs: int = 488
    save_interval: int = 50

    def __post_init__(self):
        self.global_steps = self.train_itrs * self.buffer_size
