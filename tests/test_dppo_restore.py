"""Starting a DPPO run from a checkpoint - including one FPO wrote.

DPPO had a `load_checkpoint` that nothing could reach, so a DPPO run could be
saved and never continued. The case that matters most is the cross-algorithm
one: starting DPPO from a policy FPO trained, which is how to put DPPO in the
fine-tuning setting it was designed for.

That case has a trap. FPO's checkpoint holds one optimizer; DPPO's holds an
actor's and a critic's. So `all` - a true resume - cannot take an FPO
checkpoint and must say so, while `model` and `except-critic` take its weights
and build DPPO's own optimizers fresh.
"""

from __future__ import annotations

import pathlib

import pytest
import safetensors.torch
import torch

from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

STEP = 4096
ITRS = 7


def _policy() -> FPOPolicy:
    return FPOPolicy(FPOPolicyConfig(device="cpu"))


def _dppo_config(**overrides) -> DPPOAlgoConfig:
    return DPPOAlgoConfig(
        buffer_size=64, batch_size=16, train_itrs=20, update_epochs=1, **overrides
    )


def _marked(policy: FPOPolicy) -> None:
    """Statistics no freshly built policy would have."""
    with torch.no_grad():
        policy.obs_stats_mean.fill_(3.0)
        policy.obs_stats_std.fill_(2.0)
        policy.obs_stats_count.fill_(500.0)


def _save(directory: pathlib.Path, checkpoint) -> pathlib.Path:
    manager = CheckpointManager(directory, config={}, overwrite=True, resume=False)
    manager.save_checkpoint(checkpoint)
    return directory / str(checkpoint.step)


@pytest.fixture(scope="module")
def fpo_checkpoint(tmp_path_factory) -> pathlib.Path:
    """A finished FPO run, as E16's Phase A left three of."""
    torch.manual_seed(0)
    algo = FPOAlgorithm(
        FPOAlgoConfig(global_steps=1000, buffer_size=8, batch_size=4), _policy()
    )
    for group in algo.optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.1)
    algo.optimizer.step()
    algo.master_weights.masters_to_model()
    _marked(algo.policy)
    algo.global_step = STEP
    algo.curr_train_itrs = ITRS
    return _save(tmp_path_factory.mktemp("fpo"), algo.create_checkpoint())


@pytest.fixture(scope="module")
def dppo_checkpoint(tmp_path_factory) -> pathlib.Path:
    torch.manual_seed(0)
    algo = DPPOAlgorithm(_dppo_config(), _policy())
    algo.init_optimizers()
    for optimizer, module in (
        (algo.actor_optimizer, algo.policy.actor),
        (algo.critic_optimizer, algo.policy.critic),
    ):
        for param in module.parameters():
            param.grad = torch.full_like(param, 0.1)
        optimizer.step()
    _marked(algo.policy)
    algo.global_step = STEP
    algo.curr_train_itrs = ITRS
    return _save(tmp_path_factory.mktemp("dppo"), algo.create_checkpoint())


def _restored(path: pathlib.Path, restore: str):
    torch.manual_seed(1)
    policy = _policy()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    algo = DPPOAlgorithm(
        _dppo_config(policy_checkpoint_path=path, restore=restore), policy
    )
    algo.init_optimizers()
    return algo, before


def _saved(path: pathlib.Path) -> dict[str, torch.Tensor]:
    return safetensors.torch.load_file(str(path / "model.safetensors"))


class TestResumingADppoRun:
    def test_all_brings_both_optimizers_the_step_and_the_iteration(
        self, dppo_checkpoint
    ):
        algo, _ = _restored(dppo_checkpoint, "all")

        assert algo.global_step == STEP
        assert algo.curr_train_itrs == ITRS
        assert len(algo.actor_optimizer.state) > 0
        assert len(algo.critic_optimizer.state) > 0

    def test_all_brings_every_weight(self, dppo_checkpoint):
        algo, _ = _restored(dppo_checkpoint, "all")
        saved = _saved(dppo_checkpoint)

        for key, value in algo.policy.state_dict().items():
            assert torch.equal(value, saved[key]), key


class TestStartingFromAnFpoCheckpoint:
    def test_all_is_refused_with_a_reason(self, fpo_checkpoint):
        """A resume across algorithms is not a resume. Say so, and say what to use."""
        with pytest.raises(ValueError, match="not written by DPPO"):
            _restored(fpo_checkpoint, "all")

    def test_model_takes_every_weight_and_builds_its_own_optimizers(
        self, fpo_checkpoint
    ):
        algo, _ = _restored(fpo_checkpoint, "model")
        saved = _saved(fpo_checkpoint)

        for key, value in algo.policy.state_dict().items():
            assert torch.equal(value, saved[key]), key
        assert len(algo.actor_optimizer.state) == 0
        assert len(algo.critic_optimizer.state) == 0
        assert algo.global_step == 0
        assert algo.curr_train_itrs == 0

    def test_except_critic_takes_the_actor_and_its_normalisation(self, fpo_checkpoint):
        algo, before = _restored(fpo_checkpoint, "except-critic")
        saved = _saved(fpo_checkpoint)
        now = algo.policy.state_dict()

        for key in [k for k in saved if k.startswith("actor.")]:
            assert torch.equal(now[key], saved[key]), key
        for key in ("obs_stats_mean", "obs_stats_std", "obs_stats_count"):
            assert torch.equal(now[key], saved[key]), key
            assert not torch.equal(saved[key], before[key]), (
                f"{key} is the same in the fixture and a fresh policy; proves nothing"
            )

    def test_except_critic_leaves_the_value_head_as_built(self, fpo_checkpoint):
        algo, before = _restored(fpo_checkpoint, "except-critic")
        saved = _saved(fpo_checkpoint)
        now = algo.policy.state_dict()
        critic = [k for k in saved if k.startswith("critic.")]

        assert critic
        for key in critic:
            assert torch.equal(now[key], before[key]), f"{key} was overwritten"
        assert any(not torch.equal(now[k], saved[k]) for k in critic), (
            "the fresh critic equals the saved one; the test proves nothing"
        )


class TestTheOptimizersStillPointAtTheRestoredWeights:
    """The failure that would pass every other test in this file.

    DPPO builds its optimizers before restoring. If loading rebound the
    parameters instead of copying into them, the weights would read back
    correctly and the optimizers would step tensors nothing else refers to.
    """

    @pytest.mark.parametrize("restore", ["model", "except-critic"])
    def test_a_step_after_restoring_moves_the_actor(self, fpo_checkpoint, restore):
        algo, _ = _restored(fpo_checkpoint, restore)
        before = [p.detach().clone() for p in algo.policy.actor.parameters()]
        for param in algo.policy.actor.parameters():
            param.grad = torch.full_like(param, 0.1)
        algo.actor_optimizer.step()

        assert any(
            not torch.equal(p.detach(), b)
            for p, b in zip(algo.policy.actor.parameters(), before)
        )


class TestRejections:
    def test_an_unknown_mode(self):
        with pytest.raises(ValueError, match="restore must be one of"):
            DPPOAlgoConfig(restore="half")

    def test_a_mode_without_a_checkpoint(self):
        with pytest.raises(ValueError, match="only means something"):
            DPPOAlgoConfig(restore="model")

    def test_a_missing_checkpoint(self, tmp_path):
        algo = DPPOAlgorithm(
            _dppo_config(policy_checkpoint_path=tmp_path / "nothing", restore="model"),
            _policy(),
        )
        with pytest.raises(FileNotFoundError):
            algo.init_optimizers()
