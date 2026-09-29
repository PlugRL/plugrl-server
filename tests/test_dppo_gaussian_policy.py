"""`dppo-gaussian-policy`: DPPO's Gaussian MLP, loadable from its released checkpoints.

DPPO (Ren et al. 2024, irom-lab/dppo) fine-tunes a Gaussian MLP with PPO as
the baseline on robomimic, from pretrained checkpoints it releases. For
square, `square_pre_gaussian_mlp_ta4/.../state_5000.pt` holds `model` and
`ema`, each with:

    network.mlp_mean.layers.{0,1.l1,1.l2,2}.{weight,bias}    a ResidualMLP
    network.logvar_min, network.logvar_max                   the std's bounds

and no `network.logvar`, which fine-tuning creates at log(0.1^2). The
checkpoint's `logvar_max` is log(1^2) = 0, from pretraining's default; the
fine-tuning config asks for 0.2, and DPPO's non-strict load overwrites it
with the checkpoint's. The policy keeps DPPO's behaviour and says so.

These check what DPPO's `Gaussian_MLP` and `GaussianModel` do, on a
checkpoint built to the same keys, since the released one is not in the
repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("dppo")

from plugrl_server.policy.dppo.dppo_gaussian_policy import (  # noqa: E402
    DPPOGaussianPolicy,
    DPPOGaussianPolicyConfig,
)

KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
SIZES = (3, 4, 2, 14)


def _policy(**overrides) -> DPPOGaussianPolicy:
    torch.manual_seed(0)
    return DPPOGaussianPolicy(DPPOGaussianPolicyConfig(device="cpu", **overrides))


def _obs(batch: int, rng: np.random.Generator) -> dict:
    return {
        "states": {
            k: rng.uniform(-0.3, 0.3, size=(batch, n)).astype(np.float32)
            for k, n in zip(KEYS, SIZES)
        }
    }


def _released_style_checkpoint(path, policy: DPPOGaussianPolicy) -> dict:
    """A checkpoint with the released one's keys: no logvar, logvar_max 0."""
    torch.manual_seed(1)
    donor = DPPOGaussianPolicy(DPPOGaussianPolicyConfig(device="cpu"))
    model = {
        k: v.clone()
        for k, v in donor.state_dict().items()
        if k.startswith("network.") and k != "network.logvar"
    }
    model["network.logvar_max"] = torch.tensor(0.0)
    torch.save({"epoch": 5000, "model": model, "ema": model}, path)
    return model


class TestTheNetwork:
    def test_its_keys_are_the_released_checkpoints(self):
        keys = {k for k in _policy().state_dict() if k.startswith("network.")}

        assert keys == {
            "network.logvar",
            "network.logvar_min",
            "network.logvar_max",
            *(
                f"network.mlp_mean.layers.{layer}.{p}"
                for layer in ("0", "1.l1", "1.l2", "2")
                for p in ("weight", "bias")
            ),
        }

    def test_the_mean_head_has_the_released_shapes(self):
        state = _policy().state_dict()

        assert state["network.mlp_mean.layers.0.weight"].shape == (1024, 23)
        assert state["network.mlp_mean.layers.1.l1.weight"].shape == (1024, 1024)
        assert state["network.mlp_mean.layers.2.weight"].shape == (28, 1024)
        assert state["network.logvar"].shape == (7,)

    def test_the_std_starts_at_the_fixed_std(self):
        policy = _policy()

        torch.testing.assert_close(
            policy.network.logvar.detach(), torch.full((7,), math.log(0.1**2))
        )

    def test_it_declares_its_action_shape(self):
        policy = _policy()

        assert (policy.action_dim, policy.action_horizon) == (7, 4)


class TestLoading:
    def test_the_released_keys_load_and_logvar_max_comes_with_them(self, tmp_path):
        """DPPO's load is non-strict, so the checkpoint's logvar_max (std 1.0)
        replaces the configured 0.2. Kept, so the policy behaves as DPPO's."""
        path = tmp_path / "state_5000.pt"
        model = _released_style_checkpoint(path, _policy())

        policy = _policy(checkpoint_path=path)

        for k, v in model.items():
            torch.testing.assert_close(policy.state_dict()[k], v, msg=k)
        assert policy.network.logvar_max.item() == 0.0
        torch.testing.assert_close(
            policy.network.logvar.detach(), torch.full((7,), math.log(0.1**2))
        )

    def test_it_loads_model_not_ema(self, tmp_path):
        """DPPO's GaussianModel reads `model`; its diffusion model reads `ema`."""
        path = tmp_path / "state_5000.pt"
        model = _released_style_checkpoint(path, _policy())
        ema = {k: v + 1.0 for k, v in model.items()}
        torch.save({"epoch": 5000, "model": model, "ema": ema}, path)

        policy = _policy(checkpoint_path=path)

        weight = "network.mlp_mean.layers.0.weight"
        torch.testing.assert_close(policy.state_dict()[weight], model[weight])

    def test_a_checkpoint_missing_a_mean_layer_is_refused(self, tmp_path):
        path = tmp_path / "state_5000.pt"
        model = _released_style_checkpoint(path, _policy())
        del model["network.mlp_mean.layers.2.weight"]
        torch.save({"model": model}, path)

        with pytest.raises(ValueError, match="layers.2.weight"):
            _policy(checkpoint_path=path)


class TestActing:
    def test_actions_are_chunks_in_the_environments_units(self):
        policy = _policy()
        with torch.inference_mode():
            action, state = policy.get_action_and_runtime_state(
                _obs(16, np.random.default_rng(0))
            )

        n = policy.normalization
        assert action.shape == (16, 4, 7) and action.dtype == np.float32
        assert (action >= n["action_min"] - 1e-6).all()
        assert (action <= n["action_max"] + 1e-6).all()
        assert state.action.shape == (16, 28)
        assert state.logprob.shape == state.value.shape == (16,)

    def test_observations_are_scaled_to_minus_one_one_by_the_stored_range(self):
        policy = _policy()
        obs = _obs(4, np.random.default_rng(0))

        z = policy.extract_model_obs_tensor(obs)

        n = policy.normalization
        raw = np.concatenate([obs["states"][k] for k in KEYS], axis=-1)
        expected = 2 * (raw - n["obs_min"]) / (n["obs_max"] - n["obs_min"]) - 1
        torch.testing.assert_close(z, torch.as_tensor(expected, dtype=torch.float32))

    def test_a_sample_stays_within_three_deviations_of_the_mean(self):
        policy = _policy()
        policy.network.logvar.data.fill_(0.0)  # std 1, so the clip binds often
        obs = _obs(256, np.random.default_rng(0))
        with torch.inference_mode():
            _, state = policy.get_action_and_runtime_state(obs)
            mean, scale = policy.network(state.obs)

        assert ((state.action - mean).abs() <= 3 * scale + 1e-6).all()

    def test_the_logprob_is_the_mean_over_the_chunk_clamped_to_dppos_range(self):
        policy = _policy()
        obs = _obs(8, np.random.default_rng(0))
        with torch.inference_mode():
            _, state = policy.get_action_and_runtime_state(obs)
            mean, scale = policy.network(state.obs)

        expected = torch.distributions.Normal(mean, scale).log_prob(state.action)
        torch.testing.assert_close(state.logprob, expected.mean(-1).clamp(-5.0, 2.0))

    def test_learning_scores_exactly_what_was_sampled(self):
        policy = _policy()
        with torch.inference_mode():
            _, state = policy.get_action_and_runtime_state(
                _obs(8, np.random.default_rng(0))
            )

        # Fresh tensors, as they come out of the rollout buffer.
        logprob, entropy, value = policy.evaluate_actions(
            state.obs.clone(), state.action.clone()
        )

        torch.testing.assert_close(logprob, state.logprob)
        torch.testing.assert_close(value, state.value)
        assert entropy.shape == (8,)

    def test_deterministic_acts_with_the_mean(self):
        policy = _policy(deterministic=True)
        obs = _obs(4, np.random.default_rng(0))
        with torch.inference_mode():
            _, state = policy.get_action_and_runtime_state(obs)
            mean, _ = policy.network(state.obs)

        torch.testing.assert_close(state.action, mean)
