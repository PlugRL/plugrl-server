"""FPO has to train a flow policy whose observation is a tree, not a vector.

FPO was written against FPOPolicy, whose model observation is one float
vector. Pi0Policy's is a tree: images as uint8, masks as bool, token ids as
int64, state as float. No FPO test existed, so nothing had ever run FPO on
such a policy.

TreeObsFlowPolicy below is the smallest flow policy with that observation
shape. The tests drive FPO the way the server does: infer on a batch, split
the step state per env, feed each transition back, and learn once the buffer
is full. The MLP policy runs through the same loop, so a fix for trees cannot
quietly break vectors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.policy.base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
    BasePolicyGradientFlowPolicyConfig,
)
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig
from plugrl_server.policy.state import slice_policy_step_state

IMAGE = (4, 4, 3)
TOKENS = 5
STATE_DIM = 3
ACTION_DIM = 2
HORIZON = 2
FLOW_STEPS = 3


class TreeObsFlowPolicy(BasePolicyGradientFlowPolicy):
    """A tiny flow policy whose model observation has Pi0Policy's dtypes."""

    def __init__(self) -> None:
        super().__init__(BasePolicyGradientFlowPolicyConfig(device="cpu"))
        features = int(np.prod(IMAGE)) + STATE_DIM + TOKENS + 1
        self.actor = nn.Linear(
            features + HORIZON * ACTION_DIM + 1, HORIZON * ACTION_DIM
        )
        self.critic = nn.Linear(features, 1)
        self.action_dim = ACTION_DIM
        self.action_horizon = HORIZON
        self.num_denoising_steps = FLOW_STEPS
        self.dt = -1.0 / FLOW_STEPS

    def prepare_observation(self, _obs: dict[str, Any]) -> dict[str, np.ndarray]:
        state = np.asarray(_obs["states"]["state"], dtype=np.float32)
        batch = state.shape[0]
        return dict(
            image=np.asarray(_obs["images"]["base"], dtype=np.uint8),
            image_mask=np.ones((batch,), dtype=np.bool_),
            state=state,
            tokenized_prompt=np.arange(batch * TOKENS, dtype=np.int64).reshape(
                batch, TOKENS
            ),
        )

    def fake_diffusion_cond(self, batch_size: int) -> dict[str, torch.Tensor]:
        return dict(
            image=torch.zeros((batch_size, *IMAGE), dtype=torch.uint8),
            image_mask=torch.zeros((batch_size,), dtype=torch.bool),
            state=torch.zeros((batch_size, STATE_DIM), dtype=torch.float32),
            tokenized_prompt=torch.zeros((batch_size, TOKENS), dtype=torch.long),
        )

    def build_obs_cache(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                obs["image"].float().flatten(1) / 255.0,
                obs["state"].float(),
                obs["tokenized_prompt"].float(),
                obs["image_mask"].float().reshape(-1, 1),
            ],
            dim=1,
        )

    def _features(self, cond, cond_cache, batch: int) -> torch.Tensor:
        features = cond_cache if cond_cache is not None else self.build_obs_cache(cond)
        if features.shape[0] != batch:
            features = torch.repeat_interleave(
                features, batch // features.shape[0], dim=0
            )
        return features

    def _predict_v(self, x, t, cond, *, cond_cache=None) -> torch.Tensor:
        features = self._features(cond, cond_cache, x.shape[0])
        inputs = torch.cat([features, x.flatten(1), t.reshape(-1, 1)], dim=1)
        return self.actor(inputs).reshape(x.shape)

    def _get_value(self, obs, obs_cache=None) -> torch.Tensor:
        batch = obs_cache.shape[0] if obs_cache is not None else obs["state"].shape[0]
        return self.critic(self._features(obs, obs_cache, batch)).squeeze(-1)

    def _get_timesteps(self) -> torch.Tensor:
        return torch.linspace(1.0, 1.0 / FLOW_STEPS, FLOW_STEPS)

    def _initialize_x(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, HORIZON, ACTION_DIM)

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(self, action: torch.Tensor, obs) -> np.ndarray:
        return action.detach().cpu().numpy().astype(np.float32)


def _tree_env_obs(batch: int, rng: np.random.Generator) -> dict:
    return dict(
        images=dict(base=rng.integers(0, 256, (batch, *IMAGE), dtype=np.uint8)),
        states=dict(state=rng.standard_normal((batch, STATE_DIM)).astype(np.float32)),
        text=np.asarray(["pick up the bowl"] * batch),
    )


def _vector_env_obs(batch: int, rng: np.random.Generator) -> dict:
    return dict(
        states=dict(obs=rng.standard_normal((batch, STATE_DIM)).astype(np.float32))
    )


def _config() -> FPOAlgoConfig:
    return FPOAlgoConfig(
        global_steps=10_000,
        buffer_size=8,
        batch_size=4,
        num_updates_per_batch=2,
        n_samples_per_action=2,
        learning_rate=1e-2,
    )


def _collect_and_learn(algo: FPOAlgorithm, make_obs, *, envs: int = 2) -> dict:
    """Drive the algorithm the way websocket_agent_server does."""
    rng = np.random.default_rng(0)
    prev_node = {env: (-1, "") for env in range(envs)}
    terminated = {env: False for env in range(envs)}
    obs = make_obs(envs, rng)
    for round_index in range(100):
        _, runtime_state = algo.infer(obs)
        step_state = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        next_obs = make_obs(envs, rng)
        obs_list = unbatch_aggregate(obs, aggregate_method="concat")
        next_obs_list = unbatch_aggregate(next_obs, aggregate_method="concat")
        done = round_index % 3 == 2
        for env in range(envs):
            one = slice_policy_step_state(step_state, slice(env, env + 1))
            prev_node[env], _, _ = algo.feedback(
                obs=obs_list[env],
                runtime_state=one.runtime_state,
                train_state=one.train_state,
                terminated=terminated[env],
                truncated=False,
                next_obs=next_obs_list[env],
                reward=1.0,
                info={},
                next_terminated=done,
                next_truncated=False,
                prev_node=prev_node[env],
            )
            terminated[env] = done
        obs = next_obs
        if algo.should_learn():
            break
    assert algo.should_learn(), "the buffer never filled"

    algo.pre_learn()
    _, metrics = algo.learn()
    algo.post_learn()
    return metrics


def _snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


def _finite_losses(metrics: dict) -> None:
    losses = metrics["losses"]
    for key in ("policy_loss", "value_loss", "total_loss"):
        assert np.isfinite(losses[key]), f"{key} is not finite: {losses[key]}"


class TestTreeObservations:
    def test_fpo_trains_a_policy_whose_observation_is_a_tree(self):
        torch.manual_seed(0)
        policy = TreeObsFlowPolicy()
        algo = FPOAlgorithm(_config(), policy)
        actor_before = _snapshot(policy.actor)

        metrics = _collect_and_learn(algo, _tree_env_obs)

        _finite_losses(metrics)
        assert _changed(actor_before, policy.actor), "learn left the actor untouched"

    def test_the_buffer_keeps_each_leaf_dtype(self):
        """Images stay uint8 in the buffer.

        Pi0Policy stores three 224x224x3 images per transition. As uint8 a
        4096-transition buffer holds 1.9 GB of them; cast to float32 it would
        hold 7.4 GB, before the model and its optimizer state.
        """
        torch.manual_seed(0)
        algo = FPOAlgorithm(_config(), TreeObsFlowPolicy())
        _collect_and_learn(algo, _tree_env_obs)

        stored = algo.rollout_buffer.train_state_storage.get_item(0)
        assert stored["image"].dtype == np.uint8
        assert stored["image_mask"].dtype == np.bool_
        assert stored["tokenized_prompt"].dtype == np.int64


class TestVectorObservationsStillWork:
    def test_fpo_still_trains_the_mlp_policy(self):
        torch.manual_seed(0)
        policy = FPOPolicy(
            FPOPolicyConfig(
                device="cpu",
                obs_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                action_horizon=HORIZON,
                flow_steps=FLOW_STEPS,
                hidden_dims=(8,),
                value_hidden_dims=(8,),
            )
        )
        algo = FPOAlgorithm(_config(), policy)
        actor_before = _snapshot(policy.actor)

        metrics = _collect_and_learn(algo, _vector_env_obs)

        _finite_losses(metrics)
        assert _changed(actor_before, policy.actor)
        assert algo.rollout_buffer.train_state_storage.get_item(0).dtype == np.float32


@pytest.mark.parametrize("envs", [1, 3])
def test_tree_observations_with_other_env_counts(envs):
    torch.manual_seed(0)
    algo = FPOAlgorithm(_config(), TreeObsFlowPolicy())
    _finite_losses(_collect_and_learn(algo, _tree_env_obs, envs=envs))
