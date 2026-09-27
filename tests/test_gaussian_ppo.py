"""`gaussian-policy` and `ppo`: CleanRL's continuous-action PPO, through PlugRL.

Every other policy-algorithm pair on the coverage figure trains an expressive
policy - a flow or a diffusion model - with an algorithm built for it. None is
the baseline those are measured against: a Gaussian MLP trained by PPO. These
two are that, written to CleanRL's `ppo_continuous_action.py` (Huang et al.,
"The 37 Implementation Details of Proximal Policy Optimization", 2022), so
that if the pair fails to learn MuJoCo it is the platform that failed and not
an unfamiliar algorithm.

CleanRL does part of its work in gymnasium wrappers on the environment -
clipping actions, normalising observations and rewards. PlugRL's environment
client does none of that, so here it is done on the server, and these tests
check each piece where PlugRL's plumbing could get it wrong: the sample the
log-probability was taken of is the one learned from, the ratio is one before
any update, the statistics do not move between collecting and learning, the
schedule anneals as CleanRL's does, and the whole thing improves a policy on a
task with a known answer.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.ppo import ppo as ppo_module
from plugrl_server.algorithm.ppo.ppo import PPOAlgorithm
from plugrl_server.algorithm.ppo.ppo_buffer import PPOBuffer
from plugrl_server.algorithm.ppo.ppo_config import PPOAlgoConfig
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.policy.gaussian.gaussian_policy import (
    GaussianPolicy,
    GaussianPolicyConfig,
)
from plugrl_server.policy.state import slice_policy_step_state

OBS_DIM = 5
ACTION_DIM = 3
ENVS = 2
TARGET = 0.5


def _policy(seed: int = 0, **overrides) -> GaussianPolicy:
    torch.manual_seed(seed)
    return GaussianPolicy(
        GaussianPolicyConfig(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu", **overrides
        )
    )


def _algo(seed: int = 0, **overrides) -> PPOAlgorithm:
    fields = dict(buffer_size=128, batch_size=32, update_epochs=2, train_itrs=10)
    fields.update(overrides)
    algo = PPOAlgorithm(PPOAlgoConfig(**fields), _policy(seed))
    algo.init_optimizers()
    return algo


def _obs(envs: int, rng: np.random.Generator) -> dict:
    return {"states": {"obs": rng.normal(size=(envs, OBS_DIM)).astype(np.float32)}}


def _collect(algo, rng, *, reward_fn=lambda action: 1.0, episode_length=5):
    """Fill the buffer the way websocket_agent_server does. Returns the rewards."""
    prev_node = {env: (-1, "") for env in range(ENVS)}
    terminated = {env: False for env in range(ENVS)}
    obs = _obs(ENVS, rng)
    rewards: list[float] = []
    for round_index in range(10_000):
        action, runtime_state = algo.infer(obs)
        step_state = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        next_obs = _obs(ENVS, rng)
        obs_list = unbatch_aggregate(obs, aggregate_method="concat")
        next_obs_list = unbatch_aggregate(next_obs, aggregate_method="concat")
        done = round_index % episode_length == episode_length - 1
        for env in range(ENVS):
            reward = float(reward_fn(action[env]))
            rewards.append(reward)
            one = slice_policy_step_state(step_state, slice(env, env + 1))
            prev_node[env], _, _ = algo.feedback(
                obs=obs_list[env],
                runtime_state=one.runtime_state,
                train_state=one.train_state,
                terminated=terminated[env],
                truncated=False,
                next_obs=next_obs_list[env],
                reward=reward,
                info={},
                next_terminated=done,
                next_truncated=False,
                prev_node=prev_node[env],
            )
            terminated[env] = done
        obs = next_obs
        if algo.should_learn():
            return rewards
    pytest.fail("the buffer never filled")


def _iteration(algo, rng, **collect_kwargs) -> tuple[float, dict]:
    rewards = _collect(algo, rng, **collect_kwargs)
    algo.pre_learn()
    _, metrics = algo.learn()
    algo.post_learn()
    return float(np.mean(rewards)), metrics


def _bandit_reward(action: np.ndarray) -> float:
    return -float(((action - TARGET) ** 2).mean())


class TestThePolicy:
    def test_it_declares_its_action_shape(self):
        """The server's metadata message publishes both."""
        policy = _policy()

        assert (policy.action_dim, policy.action_horizon) == (ACTION_DIM, 1)

    def test_the_env_gets_a_clipped_sample_and_the_state_keeps_the_sample(self):
        """ClipAction clips what the environment receives, not what was drawn.

        The log-probability is the density of the unclipped sample, and PPO's
        ratio is only a ratio of densities if learning scores that same sample.
        """
        policy = _policy()
        policy.actor_logstd.data.fill_(1.0)  # std e: many samples land outside
        with torch.inference_mode():
            action, state = policy.get_action_and_runtime_state(
                _obs(64, np.random.default_rng(0))
            )

        assert action.shape == (64, 1, ACTION_DIM)
        assert action.dtype == np.float32
        assert np.abs(action).max() <= 1.0
        assert state.action.shape == (64, ACTION_DIM)
        assert (state.action.abs() > 1).any()
        np.testing.assert_array_equal(action[:, 0], state.action.clamp(-1, 1).numpy())
        assert state.logprob.shape == state.value.shape == (64,)

    def test_learning_scores_exactly_what_was_sampled(self):
        policy = _policy()
        with torch.inference_mode():
            _, state = policy.get_action_and_runtime_state(
                _obs(16, np.random.default_rng(0))
            )

        logprob, entropy, value = policy.evaluate_actions(state.obs, state.action)

        torch.testing.assert_close(logprob, state.logprob)
        torch.testing.assert_close(value, state.value)
        # A diagonal Gaussian's entropy, summed over the action's dimensions.
        per_dim = 0.5 + 0.5 * math.log(2 * math.pi) + policy.actor_logstd.detach()
        torch.testing.assert_close(entropy, per_dim.sum().expand(16))

    def test_it_is_initialised_the_way_cleanrl_initialises_it(self):
        """Orthogonal weights at gain sqrt(2), 0.01 on the mean's last layer and
        1.0 on the value's, zero biases, a log std of zero, tanh between."""
        policy = _policy()

        assert torch.equal(policy.actor_logstd.detach(), torch.zeros(1, ACTION_DIM))
        for net, out, last_gain in (
            (policy.actor_mean, ACTION_DIM, 0.01),
            (policy.critic, 1, 1.0),
        ):
            layers = [m for m in net if isinstance(m, torch.nn.Linear)]
            assert [layer.out_features for layer in layers] == [64, 64, out]
            assert sum(isinstance(m, torch.nn.Tanh) for m in net) == 2
            for layer in layers:
                assert torch.equal(layer.bias.detach(), torch.zeros_like(layer.bias))
            first, hidden, last = (layer.weight.detach() for layer in layers)
            # Orthogonal: a tall matrix's columns and a wide one's rows are
            # orthonormal, times the gain.
            torch.testing.assert_close(first.T @ first, 2.0 * torch.eye(OBS_DIM))
            torch.testing.assert_close(hidden @ hidden.T, 2.0 * torch.eye(64))
            torch.testing.assert_close(
                last @ last.T, last_gain**2 * torch.eye(out), atol=1e-6, rtol=1e-5
            )

    def test_observations_are_normalised_then_clipped_at_ten(self):
        """NormalizeObservation, then TransformObservation's clip to [-10, 10]."""
        policy = _policy()
        rng = np.random.default_rng(0)
        scales = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        data = torch.as_tensor(
            rng.normal(3.0, scales, size=(4096, OBS_DIM)), dtype=torch.float32
        )

        policy.update_obs_stats(data)
        z = policy.normalize_obs(data)

        torch.testing.assert_close(z.mean(0), torch.zeros(OBS_DIM), atol=1e-3, rtol=0)
        torch.testing.assert_close(z.std(0), torch.ones(OBS_DIM), atol=1e-3, rtol=0)
        far = policy.normalize_obs(torch.full((1, OBS_DIM), 1e6))
        assert far.max().item() == 10.0

    def test_deterministic_acts_with_the_mean(self):
        """For evaluation, which acts without training's sampling noise."""
        policy = _policy(deterministic=True)
        obs = _obs(8, np.random.default_rng(0))
        with torch.inference_mode():
            first, _ = policy.get_action_and_runtime_state(obs)
            second, _ = policy.get_action_and_runtime_state(obs)
            mean = policy.actor_mean(
                policy.normalize_obs(policy.extract_model_obs_tensor(obs))
            )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first[:, 0], mean.clamp(-1, 1).numpy())

    def test_frozen_statistics_do_not_move(self):
        policy = _policy(freeze_obs_stats=True)

        policy.update_obs_stats(torch.randn(64, OBS_DIM) * 5)

        assert policy.obs_stats_count.item() == 0.0
        assert torch.equal(policy.obs_stats_std, torch.ones(OBS_DIM))

    def test_state_keys_are_concatenated_in_the_order_given(self):
        policy = _policy(state_keys=("a", "b"))
        obs = {
            "states": {
                "b": np.ones((2, 3), dtype=np.float32),
                "a": np.zeros((2, 2), dtype=np.float32),
            }
        }

        x = policy.extract_model_obs_tensor(obs)

        assert torch.equal(x, torch.tensor([[0.0, 0, 1, 1, 1]] * 2))


class TestTheBuffer:
    def _buffer(self, **kwargs) -> tuple[PPOBuffer, dict]:
        algo = _algo()
        example = algo.example_train_state(batch_size=1)
        return PPOBuffer(buffer_size=8, example_train_state=example, **kwargs), example

    def test_the_discounted_return_restarts_with_each_episode(self):
        """As gymnasium's NormalizeReward and DPPO's RunningRewardScaler keep it.

        A frame's `terminated` says the step before it ended an episode, so a
        frame carrying it is the first of a new one.
        """
        buffer, one = self._buffer(gamma=0.5)
        node = (-1, "")
        for starts_episode in (False, False, True, False):
            node = buffer.add_frame(
                prev_node=node,
                train_state=one,
                reward=1.0,
                terminated=starts_episode,
                truncated=False,
                last_value=None,
                next_terminated=False,
                next_truncated=False,
            )

        np.testing.assert_allclose(buffer.rets[:4], [1.0, 1.5, 1.0, 1.5])

    @pytest.mark.parametrize("normalize", [True, False])
    def test_rewards_are_scaled_by_the_returns_deviation_and_clipped(self, normalize):
        buffer, one = self._buffer(gamma=0.9, normalize_rewards=normalize)
        raw = np.array([0.1, 50.0, -3.0, 2.0, 0.0, 400.0, 1.0, -1.0], np.float32)
        node = (-1, "")
        for reward in raw:
            node = buffer.add_frame(
                prev_node=node,
                train_state=one,
                reward=float(reward),
                terminated=False,
                truncated=False,
                last_value=None,
                next_terminated=False,
                next_truncated=False,
            )
        buffer.add_next_obs_value_request(
            obs=_obs(1, np.random.default_rng(0)), end_node=node
        )

        buffer.compute_advantages_and_returns(policy=_policy(), batch_size=8)

        if not normalize:
            np.testing.assert_array_equal(buffer.rewards[:8], raw)
            return
        scale = np.sqrt(buffer.ret_rms.var + 1e-8)
        expected = np.clip(raw / scale, -10.0, 10.0)
        np.testing.assert_allclose(buffer.rewards[:8], expected, rtol=1e-6)
        assert np.abs(buffer.rewards[:8]).max() < np.abs(raw).max()


class TestTheAlgorithm:
    def test_one_iteration_moves_every_parameter_and_reports_finite_losses(self):
        algo = _algo()
        before = {k: v.detach().clone() for k, v in algo.policy.named_parameters()}

        _, metrics = _iteration(
            algo, np.random.default_rng(0), reward_fn=_bandit_reward
        )

        for name, parameter in algo.policy.named_parameters():
            assert not torch.equal(parameter.detach(), before[name]), name
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "old_approx_kl",
            "approx_kl",
            "clipfrac",
        ):
            assert math.isfinite(metrics["losses"][key]), key
        assert algo.curr_train_itrs == 1
        assert len(algo.rollout_buffer) == 0

    def test_before_any_update_every_ratio_is_one(self):
        algo = _algo()
        _collect(algo, np.random.default_rng(0))
        algo.pre_learn()
        n = len(algo.rollout_buffer)
        obs = torch.as_tensor(
            algo.rollout_buffer.train_state_storage.get_item(slice(0, n))
        )

        logprob, _, _ = algo.policy.evaluate_actions(
            obs, torch.as_tensor(algo.rollout_buffer.actions[:n])
        )

        torch.testing.assert_close(
            logprob, torch.as_tensor(algo.rollout_buffer.logprobs[:n])
        )

    def test_statistics_are_updated_after_learning_from_the_buffer(self):
        """After, as DPPO does for fpo-policy: one iteration collects and learns
        under one normalisation, and the next collects under the new one."""
        algo = _algo()
        rng = np.random.default_rng(0)
        _collect(algo, rng)
        algo.pre_learn()
        algo.learn()
        assert algo.policy.obs_stats_count.item() == 0.0

        algo.post_learn()

        assert algo.policy.obs_stats_count.item() == 128.0

    def test_the_learning_rate_anneals_linearly_as_cleanrl_anneals_it(self):
        """lr = (1 - (iteration - 1) / num_iterations) * learning_rate."""
        algo = _algo(train_itrs=4, learning_rate=1e-3)
        rng = np.random.default_rng(0)

        seen = [_iteration(algo, rng)[1]["models"]["learning_rate"] for _ in range(4)]

        np.testing.assert_allclose(seen, [1e-3, 0.75e-3, 0.5e-3, 0.25e-3])
        assert algo.should_stop()

    def test_without_annealing_the_rate_stays_put(self):
        algo = _algo(train_itrs=4, learning_rate=1e-3, anneal_lr=False)
        rng = np.random.default_rng(0)

        seen = [_iteration(algo, rng)[1]["models"]["learning_rate"] for _ in range(2)]

        np.testing.assert_allclose(seen, [1e-3, 1e-3])

    def test_it_refuses_a_policy_that_does_not_sample(self):
        """The ratio is of the density of what was done; a mean has none."""
        with pytest.raises(ValueError, match="deterministic"):
            PPOAlgorithm(PPOAlgoConfig(), _policy(deterministic=True))

    def test_one_adam_over_every_parameter_with_cleanrl_epsilon(self):
        algo = _algo()

        (group,) = algo.optimizer.param_groups
        assert isinstance(algo.optimizer, torch.optim.Adam)
        assert group["eps"] == 1e-5
        assert {id(p) for p in group["params"]} == {
            id(p) for p in algo.policy.parameters()
        }

    def test_every_step_clips_the_whole_gradient_at_half(self, monkeypatch):
        calls = []
        real = torch.nn.utils.clip_grad_norm_

        def recording(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            calls.append((len(parameters), max_norm))
            return real(parameters, max_norm, *args, **kwargs)

        monkeypatch.setattr(ppo_module.nn.utils, "clip_grad_norm_", recording)
        algo = _algo()
        _iteration(algo, np.random.default_rng(0))

        steps = algo.config.update_epochs * algo.config.buffer_size // 32
        assert calls == [(len(list(algo.policy.parameters())), 0.5)] * steps

    def test_a_checkpoint_resumes_where_it_left_off(self):
        algo = _algo()
        rng = np.random.default_rng(0)
        _iteration(algo, rng, reward_fn=_bandit_reward)
        checkpoint = algo.create_checkpoint()

        resumed = _algo(seed=1)
        resumed.load_checkpoint(checkpoint)

        for (name, a), b in zip(
            algo.policy.state_dict().items(), resumed.policy.state_dict().values()
        ):
            assert torch.equal(a, b), name
        assert resumed.curr_train_itrs == 1
        assert resumed.global_step == algo.global_step
        assert resumed.rollout_buffer.ret_rms.var == algo.rollout_buffer.ret_rms.var
        assert resumed.rollout_buffer.ret_rms.count == algo.rollout_buffer.ret_rms.count
        state, other = algo.optimizer.state_dict(), resumed.optimizer.state_dict()
        for key in state["state"]:
            assert torch.equal(
                state["state"][key]["exp_avg"], other["state"][key]["exp_avg"]
            )
        # The schedule is a function of the iteration, so it resumes with it.
        _, metrics = _iteration(resumed, rng)
        assert metrics["models"]["learning_rate"] == pytest.approx(3e-4 * 0.9)


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4))
def test_ppo_learns_a_bandit_with_a_known_answer_and_keeps_it(seed: int):
    """The bandit `test_fpo_learns_anything` gives FPO: reward is minus the
    squared distance from the action to 0.5 in every dimension.

    A Gaussian at mean zero with unit deviation scores about -0.78 here. It
    has to end at a quarter of that, and no worse than twice its best - the
    bar `test_fpo_holds_what_it_learns` sets FPO, and which some of FPO's
    cases fail. Measured, last over best: 1.00, 1.14, 1.00, 1.00, 1.03,
    ending between -0.120 and -0.160.

    The rate is 3e-3, not CleanRL's 3e-4, because the floor on this reward is
    set by how far the deviation narrows, and at 3e-4, annealed over thirty
    iterations of this size, it cannot narrow far: seed 0 goes from -0.78 to
    -0.23, still rising at the last iteration.
    """
    algo = _algo(
        seed=seed,
        buffer_size=256,
        batch_size=64,
        update_epochs=10,
        train_itrs=30,
        learning_rate=3e-3,
    )
    rng = np.random.default_rng(seed)

    history = [_iteration(algo, rng, reward_fn=_bandit_reward)[0] for _ in range(30)]

    best, last = max(history), history[-1]
    assert np.isfinite(history).all(), history
    assert last > history[0] / 4, (
        f"first {history[0]:+.4f}, best {best:+.4f}, last {last:+.4f}"
    )
    assert last > 2 * best, f"best {best:+.4f}, last {last:+.4f}"


def test_both_are_on_the_cli_of_a_plain_install():
    """In a fresh interpreter, so it is the packages' own imports that register
    them and not this file's."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plugrl_server.cli",
            "gaussian-policy",
            "default",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "ppo" in result.stdout
