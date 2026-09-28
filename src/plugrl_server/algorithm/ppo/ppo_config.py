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
    # None: no clipping (DPPO's Gaussian PPO).
    max_grad_norm: float | None = 0.5
    # Adam's epsilon: CleanRL's 1e-5. DPPO keeps torch's 1e-8.
    adam_eps: float = 1e-5
    # A learning rate of the critic's own, in a parameter group of its own;
    # None keeps CleanRL's one rate for actor and critic. Annealing scales
    # both.
    critic_learning_rate: float | None = None
    # Iterations at the start in which only the critic learns (DPPO: 1 on
    # square, after its evaluation-only iteration 0).
    n_critic_warmup_itrs: int = 0
    # Stop an iteration's epochs once the approximate KL passes this. None,
    # CleanRL's default, never stops.
    target_kl: float | None = None
    # gymnasium's NormalizeReward and TransformReward, which CleanRL wraps its
    # environments in: rewards divided by the running deviation of the
    # discounted return, then clipped.
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    # The discount of the running return that rewards are scaled by, when it
    # is not `gamma`. DPPO's RunningRewardScaler keeps its own 0.99 while its
    # GAE discounts by 0.999.
    reward_scaling_gamma: float | None = None
    train_itrs: int = 488
    save_interval: int = 50

    def __post_init__(self):
        self.global_steps = self.train_itrs * self.buffer_size


@register_algo_config(UID, "dppo-square")
@dataclasses.dataclass
class PPOAlgoConfigDPPOSquare(PPOAlgoConfig):
    """DPPO's Gaussian-policy PPO on robomimic square, state input.

    `cfg/robomimic/finetune/square/ft_ppo_gaussian_mlp.yaml` at irom-lab/dppo
    cc7234ad, for `dppo-gaussian-policy` started from DPPO's released
    checkpoint. One iteration is 50 environments x 400 action chunks = 20,000
    chunks (80,000 steps), two minibatches of 10,000 over ten epochs. The
    rates are the current config's: 1e-4 and 1e-3, constant (the paper's
    table lists an actor rate of 1e-5, which the config used before v0.7).

    Not carried over: DPPO's deterministic evaluation every tenth iteration,
    which trains nothing; and its GAE, which masks only termination and so,
    on square, where episodes only time out, bootstraps across resets. Here a
    time-out ends the episode for GAE, as for every other algorithm.
    """

    learning_rate: float = 1e-4
    critic_learning_rate: float | None = 1e-3
    anneal_lr: bool = False
    buffer_size: int = 20000
    batch_size: int = 10000
    update_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.01
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float | None = None
    target_kl: float | None = 1.0
    adam_eps: float = 1e-8
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    reward_scaling_gamma: float | None = 0.99
    n_critic_warmup_itrs: int = 1
    train_itrs: int = 40
    save_interval: int = 10
