"""`dppo-policy` can fine-tune only its last denoising steps, as DPPO does on robomimic.

DPPO's robomimic configs fine-tune the last 10 of 20 denoising steps: the
first 10 run a frozen copy of the pretrained network, and only the last 10
are PPO samples (`ft_denoising_steps` in DPPO's VPGDiffusion). `dppo-policy`
had one network for all 20 and made every step a sample. With
`ft_denoising_steps` set, the early steps run a frozen copy taken after the
checkpoint is loaded, every step's deviation keeps its floor, and the policy
records only the fine-tuned steps, which is what DPPO's loss then sees.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("dppo")

from plugrl_server.policy.dppo.dppo_policy import DPPOPolicy, DPPOPolicyConfig  # noqa: E402

B = 3
KEYS = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "object": 14,
}


def _policy(ft: int | None = None, checkpoint=None) -> DPPOPolicy:
    torch.manual_seed(0)
    return DPPOPolicy(
        DPPOPolicyConfig(
            device="cpu",
            env_type="robomimic",
            env_name="square",
            ft_denoising_steps=ft,
            checkpoint_path=checkpoint,
        )
    )


def _obs() -> dict:
    rng = np.random.default_rng(0)
    return {"states": {k: rng.uniform(-1, 1, (B, d)) for k, d in KEYS.items()}}


def test_by_default_every_step_is_a_sample():
    policy = _policy()
    with torch.no_grad():
        _, state = policy.get_action_and_runtime_state(_obs(), sampling_noise_level=0.1)
    assert state.action.shape[1] == 20
    assert state.obs.t[0].tolist() == list(range(19, -1, -1))


def test_only_the_fine_tuned_steps_are_recorded():
    policy = _policy(ft=10)
    with torch.no_grad():
        _, state = policy.get_action_and_runtime_state(_obs(), sampling_noise_level=0.1)
    for leaf in (state.action, state.logprob, state.entropy, state.obs.x):
        assert leaf.shape[:2] == (B, 10)
    assert state.obs.t[0].tolist() == list(range(9, -1, -1))


def test_the_early_steps_run_a_frozen_copy():
    policy = _policy(ft=10)
    assert not any(p.requires_grad for p in policy.actor_frozen.parameters())
    x = torch.randn(B, policy.action_horizon, policy.action_dim)
    cond = {"state": torch.randn(B, policy.obs_dim)}

    def means(t: int) -> torch.Tensor:
        torch.manual_seed(1)
        nxt, _, _ = policy._denoising_step(x, torch.full((B,), t), cond)
        return nxt

    early, late = means(15), means(3)
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(0.1)
    assert torch.equal(means(15), early), "a frozen step moved with the actor"
    assert not torch.equal(means(3), late), "a fine-tuned step ignored the actor"


def test_the_frozen_copy_is_the_loaded_checkpoint(tmp_path):
    state = {k: v + 1.0 for k, v in _policy().actor.state_dict().items()}
    path = tmp_path / "state.pt"
    torch.save({"ema": state}, path)
    frozen = _policy(ft=10, checkpoint=path).actor_frozen.state_dict()
    assert all(torch.equal(frozen[k], state[k]) for k in state)


def test_a_batch_of_recorded_steps_repeats_its_condition_per_step():
    """DPPO's loss passes (batch x fine-tuned steps) states against a batch of conditions."""
    policy = _policy(ft=10)
    x = torch.randn(B * 10, policy.action_horizon, policy.action_dim)
    t = torch.arange(9, -1, -1).repeat(B)
    cond = {"state": torch.randn(B, policy.obs_dim)}
    _, logprob, _ = policy._denoising_step(
        x, t, cond, x_next=x, sampling_noise_level=0.1
    )
    assert logprob.shape == x.shape


def test_dppo_learns_through_the_fine_tuned_steps_only():
    """End to end: the trained network moves, the frozen copy does not."""
    from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
    from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
    from plugrl_server.common.data_utils import unbatch_aggregate
    from plugrl_server.policy.state import slice_policy_step_state

    policy = _policy(ft=10)
    config = DPPOAlgoConfig(
        buffer_size=16, batch_size=8, update_epochs=1, train_itrs=20, grad_accum_steps=1
    )
    algo = DPPOAlgorithm(config, policy)
    algo.init_optimizers()
    trained = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    frozen = {k: v.clone() for k, v in policy.actor_frozen.state_dict().items()}

    rng = np.random.default_rng(0)

    def obs(envs: int) -> dict:
        return {"states": {k: rng.uniform(-1, 1, (envs, d)) for k, d in KEYS.items()}}

    envs, prev = 2, {0: (-1, ""), 1: (-1, "")}
    current = obs(envs)
    for round_index in range(100):
        _, runtime_state = algo.infer(current)
        step = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        following = obs(envs)
        now = unbatch_aggregate(current, aggregate_method="concat")
        after = unbatch_aggregate(following, aggregate_method="concat")
        for env in range(envs):
            one = slice_policy_step_state(step, slice(env, env + 1))
            prev[env], _, _ = algo.feedback(
                obs=now[env], runtime_state=one.runtime_state,
                train_state=one.train_state, terminated=False, truncated=False,
                next_obs=after[env], reward=float(rng.normal()), info={},
                next_terminated=round_index % 4 == 3, next_truncated=False,
                prev_node=prev[env],
            )  # fmt: skip
        current = following
        if algo.should_learn():
            break
    assert algo.should_learn(), "the buffer never filled"

    algo.pre_learn()
    algo.learn()
    algo.post_learn()
    after_trained = policy.actor.state_dict()
    after_frozen = policy.actor_frozen.state_dict()
    assert any(not torch.equal(trained[k], after_trained[k]) for k in trained)
    assert all(torch.equal(frozen[k], after_frozen[k]) for k in frozen)


@pytest.mark.parametrize("ft", [0, 21])
def test_a_count_outside_the_chain_is_refused(ft):
    with pytest.raises(ValueError, match="ft_denoising_steps"):
        _policy(ft=ft)


def test_it_parses_from_the_command_line():
    import tyro

    config = tyro.cli(DPPOPolicyConfig, args=["--ft-denoising-steps", "10"])
    assert config.ft_denoising_steps == 10
