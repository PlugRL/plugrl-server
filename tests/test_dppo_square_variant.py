"""`dppo-policy square dppo square` is DPPO's own robomimic square fine-tuning.

Every value is the one in DPPO's
`cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp.yaml`. Where PlugRL counts
differently, the test says how the two correspond: PlugRL's buffer holds one
entry per action chunk (DPPO's 50 envs x 400 steps), and a batch holds
entries of 10 fine-tuned denoising steps (DPPO's 10,000 samples).
"""

from __future__ import annotations

import pytest

from plugrl_server.algorithm.registration import REGISTERED_ALGO_CONFIGS


@pytest.fixture
def algo():
    return REGISTERED_ALGO_CONFIGS["dppo"]["square"]


def test_the_algorithm_variant_is_dppos_square_config(algo):
    assert (algo.gamma, algo.gamma_denoising, algo.gae_lambda) == (0.999, 0.99, 0.95)
    assert (algo.actor_lr, algo.critic_lr) == (1e-4, 1e-3)
    assert (algo.actor_weight_decay, algo.critic_weight_decay) == (0.0, 0.0)
    assert algo.update_epochs == 10
    assert algo.vf_coef == 0.5
    assert algo.target_kl == 1.0
    assert algo.norm_adv
    assert (algo.clip_ploss_coef, algo.clip_ploss_coef_base) == (0.01, 0.001)
    assert algo.clip_ploss_coef_rate == 3
    assert (algo.sampling_noise_level, algo.logprob_noise_level) == (0.1, 0.1)
    assert algo.n_critic_warmup_itrs == 2
    assert algo.use_normalized_rewards
    assert algo.train_itrs == 201


def test_its_learning_rates_are_constant(algo):
    """DPPO's schedulers run from each rate to a minimum equal to it."""
    assert algo.actor_lr_scheduler is None and algo.critic_lr_scheduler is None


def test_an_iteration_and_a_batch_are_dppos_in_plugrls_units(algo):
    assert algo.buffer_size == 50 * 400  # chunks: 80,000 environment steps
    assert algo.batch_size * 10 == 10_000  # entries x fine-tuned steps
    assert algo.grad_accum_steps == 1  # one optimizer step per minibatch


def test_the_policy_variant_fine_tunes_the_last_ten_of_twenty_steps():
    pytest.importorskip("dppo")  # dppo-policy registers only with the extra
    from plugrl_server.policy.dppo.dppo_policy import DPPOPolicyConfigSquare

    policy = DPPOPolicyConfigSquare()
    assert (policy.env_type, policy.env_name) == ("robomimic", "square")
    assert policy.ft_denoising_steps == 10
    assert policy.checkpoint_path is None  # the released checkpoint is passed in
