"""Starting an FPO run from a checkpoint, and choosing how much of one to take.

`FPOAlgorithm.load_checkpoint` has always restored model, optimizer, step and
iteration count. Nothing ever called it: `policy_checkpoint_path` existed only
on `eval`, so a training run could be saved and never continued. E6 trained
HalfCheetah for 500,000 steps across three seeds and those checkpoints could
only be evaluated.

Three modes, because resuming and starting-from-weights are different things:

  all            a true resume - the optimizer's moments come too.
  model          weights only; the optimizer starts over.
  except-critic  everything but `critic.*`, so the value head keeps its random
                 initialisation.

`except-critic` is E14's shape written down. A pi0.5 arrived pretrained and
carrying its own normalisation, met a value head that had never seen it, and
lost 29 of 50 in one iteration. Reproducing that shape on a model small enough
to debug needs a way to build it, which is what these three modes are for.

The last test in `TestExceptCritic` is the one that makes the comparison mean
anything: `obs_stats_*` sit outside both `actor.` and `critic.`, and dropping
them would feed the restored actor observations normalised differently from
the ones it learned on. That is a second change, not the one under test.
"""

from __future__ import annotations

import pathlib

import pytest
import safetensors.torch
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig


STEP = 4096
ITRS = 7


def _config() -> FPOAlgoConfig:
    return FPOAlgoConfig(
        global_steps=1_000,
        buffer_size=8,
        batch_size=4,
        num_updates_per_batch=2,
        n_samples_per_action=2,
        learning_rate=1e-2,
    )


def _policy() -> FPOPolicy:
    """The policy E6 trained: 272k parameters, actor and critic, CPU.

    Not one of the stand-ins in the other test modules, because two of the
    four tensors this file is about - `obs_stats_*` - exist only on the real
    one, and they are the reason `except-critic` is not `actor`-only.
    """
    return FPOPolicy(FPOPolicyConfig(device="cpu"))


def _saved_run(directory: pathlib.Path) -> pathlib.Path:
    """A finished run on disk: moved weights, real optimizer state, a step."""
    torch.manual_seed(0)
    algo = FPOAlgorithm(_config(), _policy())

    # Move every parameter and give Adam two non-empty moments, without
    # driving a whole collection loop for it.
    for group in algo.optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.1)
    algo.optimizer.step()
    algo.master_weights.masters_to_model()

    # Observation statistics that no fresh policy would have.
    with torch.no_grad():
        algo.policy.obs_stats_mean.fill_(3.0)
        algo.policy.obs_stats_std.fill_(2.0)
        algo.policy.obs_stats_count.fill_(500.0)

    algo.global_step = STEP
    algo.curr_train_itrs = ITRS

    manager = CheckpointManager(directory, config={}, overwrite=True, resume=False)
    manager.save_checkpoint(algo.create_checkpoint())
    return directory / str(STEP)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> pathlib.Path:
    return _saved_run(tmp_path_factory.mktemp("run"))


def _restored(checkpoint: pathlib.Path, restore: str):
    """A fresh policy, snapshotted before the algorithm may overwrite it."""
    torch.manual_seed(1)
    policy = _policy()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    config = _config()
    config.policy_checkpoint_path = checkpoint
    config.restore = restore
    config.__post_init__()
    return FPOAlgorithm(config, policy), before


def _saved_state(checkpoint: pathlib.Path) -> dict[str, torch.Tensor]:
    return safetensors.torch.load_file(checkpoint / "model.safetensors")


class TestAll:
    def test_a_true_resume_brings_the_optimizer_and_the_step(self, checkpoint):
        algo, _ = _restored(checkpoint, "all")

        assert algo.global_step == STEP
        assert algo.curr_train_itrs == ITRS
        assert len(algo.optimizer.state) > 0

    def test_a_true_resume_brings_every_weight(self, checkpoint):
        algo, _ = _restored(checkpoint, "all")
        saved = _saved_state(checkpoint)

        for key, value in algo.policy.state_dict().items():
            assert torch.equal(value, saved[key]), key


class TestModel:
    def test_weights_arrive_but_the_optimizer_starts_over(self, checkpoint):
        algo, _ = _restored(checkpoint, "model")
        saved = _saved_state(checkpoint)

        for key, value in algo.policy.state_dict().items():
            assert torch.equal(value, saved[key]), key
        assert len(algo.optimizer.state) == 0
        assert algo.global_step == 0
        assert algo.curr_train_itrs == 0


class TestExceptCritic:
    def test_the_actor_arrives(self, checkpoint):
        algo, _ = _restored(checkpoint, "except-critic")
        saved = _saved_state(checkpoint)

        actor = [k for k in saved if k.startswith("actor.")]
        assert actor, "the fixture has no actor tensors to check"
        for key in actor:
            assert torch.equal(algo.policy.state_dict()[key], saved[key]), key

    def test_the_critic_keeps_its_own_initialisation(self, checkpoint):
        algo, before = _restored(checkpoint, "except-critic")
        saved = _saved_state(checkpoint)

        critic = [k for k in saved if k.startswith("critic.")]
        assert critic, "the fixture has no critic tensors to check"
        now = algo.policy.state_dict()
        for key in critic:
            assert torch.equal(now[key], before[key]), f"{key} was overwritten"
        assert any(
            not torch.equal(now[key], saved[key]) for key in critic
        ), "the fresh critic happens to equal the saved one; the test proves nothing"

    def test_normalisation_comes_with_the_actor(self, checkpoint):
        """The guard against measuring the wrong change.

        `obs_stats_*` are not under `critic.`, so they are restored. Were they
        not, the actor would see differently normalised observations and any
        collapse would have two candidate causes instead of one.
        """
        algo, before = _restored(checkpoint, "except-critic")
        saved = _saved_state(checkpoint)

        for key in ("obs_stats_mean", "obs_stats_std", "obs_stats_count"):
            assert torch.equal(algo.policy.state_dict()[key], saved[key]), key
            assert not torch.equal(
                saved[key], before[key]
            ), f"{key} is the same before and after; the test proves nothing"

    def test_the_optimizer_starts_over(self, checkpoint):
        algo, _ = _restored(checkpoint, "except-critic")

        assert len(algo.optimizer.state) == 0
        assert algo.global_step == 0


class TestRejections:
    def test_an_unknown_restore_mode_is_refused(self):
        with pytest.raises(ValueError, match="restore must be one of"):
            FPOAlgoConfig(restore="half")

    def test_a_restore_mode_without_a_checkpoint_is_refused(self):
        with pytest.raises(ValueError, match="only means something"):
            FPOAlgoConfig(restore="model")

    def test_a_missing_checkpoint_is_refused(self, tmp_path):
        config = _config()
        config.policy_checkpoint_path = tmp_path / "nothing-here"
        config.restore = "model"
        with pytest.raises(FileNotFoundError):
            FPOAlgorithm(config, _policy())


def _restored_bf16(checkpoint: pathlib.Path, restore: str):
    """The same restore, into a policy whose actor is half precision.

    `FPOPolicy` is float32 throughout, so `MasterWeights` builds no copies for
    it - zero pairs - and `sync_from_model` iterates nothing. Casting the actor
    to bfloat16 gives ten pairs, which is what pi0.5's action expert looks like
    and the only way these tests reach that path at all.
    """
    torch.manual_seed(1)
    policy = _policy()
    policy.actor = policy.actor.to(torch.bfloat16)
    config = _config()
    config.policy_checkpoint_path = checkpoint
    config.restore = restore
    config.__post_init__()
    return FPOAlgorithm(config, policy)


def _steps_and_moves(algo) -> bool:
    """One optimizer step through the real path; did the policy change?"""
    before = {k: v.clone() for k, v in algo.policy.state_dict().items()}
    algo.master_weights.clear_model_grads()
    for group in algo.optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.1)
    algo.optimizer.step()
    algo.master_weights.masters_to_model()
    now = algo.policy.state_dict()
    return any(
        not torch.equal(now[key], before[key])
        for key in before
        if now[key].is_floating_point()
    )


class TestTheOptimizerStillPointsAtTheRestoredWeights:
    """The silent failure this guards against.

    The optimizer is built in `__init__`, from `master_weights.optimizer_params`,
    before anything is restored. Restoring then writes into the policy and, for
    half-precision parameters, into the master copies. That only works because
    both writes are in place - `nn.Module.load_state_dict` copies into
    `param.data`, and `MasterWeights.sync_from_model` does
    `master.copy_(param)`.

    Had either rebound its tensors instead, every other test in this file
    would still pass: the weights would read back correctly, the step count
    would be right, and the optimizer would be stepping objects nothing else
    refers to. Training would run, log sensible losses, and change nothing.

    So these take an actual optimizer step rather than comparing identities.
    """

    def test_a_step_after_a_full_resume_moves_the_policy(self, checkpoint):
        algo, _ = _restored(checkpoint, "all")

        assert _steps_and_moves(algo)

    def test_a_step_after_a_weights_only_restore_moves_the_policy(self, checkpoint):
        algo, _ = _restored(checkpoint, "model")

        assert _steps_and_moves(algo)

    def test_a_step_after_except_critic_moves_the_policy(self, checkpoint):
        algo, _ = _restored(checkpoint, "except-critic")

        assert _steps_and_moves(algo)


class TestHalfPrecisionGoesThroughTheMasterCopies:
    """The three above cannot reach `sync_from_model`; these can.

    The checkpoint holds float32 weights and the actor here is bfloat16, so
    the restored values are rounded on the way in. That is why these assert
    movement rather than equality - the equality tests elsewhere in this file
    would be measuring the cast, not the restore.
    """

    def test_the_actor_has_master_copies_at_all(self, checkpoint):
        """Without this the two below would pass against an empty loop."""
        algo = _restored_bf16(checkpoint, "all")

        assert len(algo.master_weights.pairs) > 0

    def test_a_step_after_a_full_resume_moves_a_bfloat16_actor(self, checkpoint):
        algo = _restored_bf16(checkpoint, "all")

        assert _steps_and_moves(algo)

    def test_a_step_after_except_critic_moves_a_bfloat16_actor(self, checkpoint):
        algo = _restored_bf16(checkpoint, "except-critic")

        assert _steps_and_moves(algo)
