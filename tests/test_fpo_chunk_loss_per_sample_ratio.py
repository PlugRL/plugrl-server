"""FPO++'s way of scoring an action chunk, as options (E32).

FPO++ (Yi, Choi et al. 2026; amazon-far/fpo-control) fine-tunes pretrained
flow policies with two changes this FPO lacked:

* **the chunk's loss** - the velocity error's mean over the action
  dimensions, summed over the chunk's steps, where this FPO averaged every
  element of the chunk. For pi0.5 on LIBERO that average ran over 10 steps of
  32 dimensions, of which the environment uses 7 and the client executes 5
  steps: 89% of what the ratio averaged was padding or never happened.
  `cfm_loss_steps` and `cfm_loss_dims` keep only the executed steps and the
  used dimensions; `cfm_loss_sum_over_steps` sums over steps.
* **one ratio per Monte Carlo sample**, each clipped on its own, where this
  FPO averaged the samples' losses into one ratio per action. The log-ratio
  is clamped straight-through at +-5, as the reference does.

Every option defaults to off, and off is FPO's behaviour until now.
"""

from __future__ import annotations

import pytest
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_buffer import FPOBuffer
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.algorithm.fpo.utils import ChunkReduction, compute_cfm_loss
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

B, S, H, D = 16, 4, 4, 6


class _ZeroVelocity:
    """A policy predicting velocity 0, so the u-mode error is (a - eps)^2 exactly."""

    def _predict_v(self, x_t, t, obs, cond_cache=None):
        return torch.zeros_like(x_t)


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    action = torch.randn(B, H, D, generator=g)
    loss_eps = torch.randn(B, S, H, D, generator=g)
    loss_t = torch.rand(B, S, 1, generator=g)
    return action, loss_eps, loss_t


def _loss(reduction: ChunkReduction | None = None, output_mode: str = "u"):
    action, loss_eps, loss_t = _inputs()
    kwargs = {} if reduction is None else {"reduction": reduction}
    return compute_cfm_loss(
        _ZeroVelocity(), None, action,
        output_mode=output_mode, loss_eps=loss_eps, loss_t=loss_t, **kwargs,
    )  # fmt: skip


def _error() -> torch.Tensor:
    action, loss_eps, _ = _inputs()
    return (action.unsqueeze(1) - loss_eps).pow(2)  # (B, S, H, D)


class TestTheChunkLoss:
    def test_the_default_averages_every_element(self):
        assert torch.allclose(_loss(), _error().reshape(B, S, -1).mean(-1))

    def test_the_default_reduction_is_the_old_behaviour_in_eps_mode_too(self):
        assert torch.equal(
            _loss(output_mode="u_but_supervise_as_eps"),
            _loss(ChunkReduction(), output_mode="u_but_supervise_as_eps"),
        )

    def test_only_the_executed_steps_and_used_dims(self):
        got = _loss(ChunkReduction(steps=2, dims=3))
        assert torch.allclose(got, _error()[:, :, :2, :3].mean(-1).mean(-1))

    def test_fpo_plus_plus_sums_the_steps_of_a_mean_over_dims(self):
        got = _loss(ChunkReduction(steps=3, dims=5, sum_over_steps=True))
        assert torch.allclose(got, _error()[:, :, :3, :5].mean(-1).sum(-1))

    def test_more_steps_or_dims_than_the_chunk_has_is_refused(self):
        with pytest.raises(ValueError, match="steps"):
            _loss(ChunkReduction(steps=H + 1))
        with pytest.raises(ValueError, match="dims"):
            _loss(ChunkReduction(dims=D + 1))


def _policy() -> FPOPolicy:
    torch.manual_seed(0)
    return FPOPolicy(FPOPolicyConfig(device="cpu", action_horizon=H, action_dim=D))


def _algo(**overrides) -> FPOAlgorithm:
    config = FPOAlgoConfig(buffer_size=B, batch_size=B, output_mode="u", **overrides)
    return FPOAlgorithm(config, _policy())


def test_the_algorithm_reads_the_reduction_from_its_config():
    algo = _algo(cfm_loss_steps=3, cfm_loss_dims=5, cfm_loss_sum_over_steps=True)
    assert algo.chunk_reduction == ChunkReduction(3, 5, True)
    assert _algo().chunk_reduction == ChunkReduction()


def test_the_buffers_initial_loss_uses_the_same_reduction():
    """The ratio compares two losses; both must be the same kind of loss."""
    algo = _algo()
    buffer = FPOBuffer(
        buffer_size=B,
        example_train_state=algo.example_train_state(batch_size=1),
        n_samples_per_action=S,
    )
    obs = torch.randn(4, algo.policy.config.obs_dim)
    buffer.train_state_storage.set_item(slice(0, 4), obs.numpy())
    buffer.actions[:4] = torch.randn(4, H, D).numpy()
    buffer.idx = 4
    reduction = ChunkReduction(steps=2, dims=3, sum_over_steps=True)
    buffer.prepare_fpo_fields(
        algo.policy, batch_size=4, output_mode="u", reduction=reduction
    )
    expected = compute_cfm_loss(
        algo.policy, obs, torch.from_numpy(buffer.actions[:4]),
        output_mode="u", loss_eps=torch.from_numpy(buffer.loss_eps[:4]),
        loss_t=torch.from_numpy(buffer.loss_t[:4]),
        obs_cache=algo.policy.build_obs_cache(obs), reduction=reduction,
    )  # fmt: skip
    assert torch.allclose(torch.from_numpy(buffer.initial_cfm_loss[:4]), expected)


def test_pre_learn_passes_the_reduction_to_the_buffer():
    algo = _algo(cfm_loss_steps=2, cfm_loss_dims=3, cfm_loss_sum_over_steps=True)
    seen = {}

    class Seen(Exception):
        pass

    def prepare(policy, **kwargs):
        seen.update(kwargs)
        raise Seen

    algo.rollout_buffer.prepare_fpo_fields = prepare
    with pytest.raises(Seen):
        algo.pre_learn()
    assert seen["reduction"] == ChunkReduction(2, 3, True)


def _step(algo: FPOAlgorithm, advantage: torch.Tensor, shift: float = 0.0):
    """One _compute_loss on fixed inputs; the stored loss is the current one plus `shift`."""
    action, loss_eps, loss_t = _inputs(1)
    obs = torch.randn(B, algo.policy.config.obs_dim)
    with torch.no_grad():
        current = compute_cfm_loss(
            algo.policy, obs, action, output_mode="u", loss_eps=loss_eps,
            loss_t=loss_t, obs_cache=algo.policy.build_obs_cache(obs),
            reduction=algo.chunk_reduction,
        )  # fmt: skip
    initial = current + shift * torch.linspace(-1, 1, S).expand(B, S)
    zeros = torch.zeros(B)
    algo.policy.actor.zero_grad()
    loss, metrics, _ = algo._compute_loss(
        obs, action, zeros, zeros, advantage, zeros, loss_eps, loss_t, initial
    )
    loss.backward()
    grad = sum(
        float(p.grad.pow(2).sum())
        for p in algo.policy.actor.parameters()
        if p.grad is not None
    )
    return metrics, grad, current, initial


class TestThePerSampleRatio:
    def test_off_by_default(self):
        assert FPOAlgoConfig().ratio_per_sample is False

    def test_each_sample_is_clipped_on_its_own(self):
        """The policy loss is the mean over samples of each sample's PPO term."""
        eps = 0.05
        algo = _algo(ratio_per_sample=True, clipping_epsilon=eps)
        advantage = torch.linspace(-1.5, 1.5, B)
        metrics, _, current, initial = _step(algo, advantage, shift=0.2)
        rho = torch.exp(initial - current)
        a = advantage.reshape(B, 1)
        expected = -torch.minimum(rho * a, rho.clamp(1 - eps, 1 + eps) * a).mean()
        assert metrics["policy_loss"] == pytest.approx(float(expected), rel=1e-5)

    def test_it_differs_from_one_ratio_per_action(self):
        advantage = torch.linspace(-1.5, 1.5, B)
        per_sample, _, _, _ = _step(_algo(ratio_per_sample=True), advantage, 0.2)
        per_action, _, _, _ = _step(_algo(), advantage, 0.2)
        assert per_sample["policy_loss"] != pytest.approx(per_action["policy_loss"])

    def test_the_log_ratio_clamp_passes_the_gradient_through(self):
        """At a log-ratio of +10 a negative advantage still pushes, as with clamp_ste.

        A plain clamp would zero the gradient there; the per-action path's
        clamp at +-3 does.
        """
        advantage = -torch.ones(B)
        metrics, grad, _, _ = _step(_algo(ratio_per_sample=True), advantage, 10.0)
        assert grad > 0.0
        assert metrics["policy_ratio_mean"] <= float(torch.exp(torch.tensor(5.0)))


def test_fpo_plus_plus_trains_a_policy_whose_observation_is_a_tree():
    """End to end on the stand-in with pi0-policy's observation, every option on."""
    import dataclasses

    from test_fpo_tree_observations import (
        TreeObsFlowPolicy,
        _changed,
        _collect_and_learn,
        _config,
        _finite_losses,
        _snapshot,
        _tree_env_obs,
    )

    torch.manual_seed(0)
    policy = TreeObsFlowPolicy()
    config = dataclasses.replace(
        _config(),
        output_mode="u",
        discretize_t_for_training=False,
        cfm_loss_steps=1,
        cfm_loss_dims=1,
        cfm_loss_sum_over_steps=True,
        ratio_per_sample=True,
    )
    algo = FPOAlgorithm(config, policy)
    before = _snapshot(policy.actor)
    metrics = _collect_and_learn(algo, _tree_env_obs)
    _finite_losses(metrics)
    assert _changed(before, policy.actor)


def test_the_options_parse_from_the_command_line():
    import tyro

    config = tyro.cli(
        FPOAlgoConfig,
        args=[
            "--cfm-loss-steps", "5", "--cfm-loss-dims", "7",
            "--cfm-loss-sum-over-steps", "--ratio-per-sample",
        ],
    )  # fmt: skip
    assert (config.cfm_loss_steps, config.cfm_loss_dims) == (5, 7)
    assert config.cfm_loss_sum_over_steps and config.ratio_per_sample
