"""What moves first when the policy starts losing what it learned?

Eight hypotheses have been refuted by experiment, the last two here: there is
no dtype effect, and advantage normalisation is load-bearing rather than the
cause - turning it off stops the policy learning at all and blew one run up to
-9.3e6.

So stop proposing causes and look at what the algorithm reports about itself.
Per iteration this records the reward alongside FPO's own metrics, for a seed
where the policy comes apart and one where it holds. Whatever turns before
the reward does is a candidate trigger, and a quantity that predicts the turn
is something an early stop could watch - which standard PPO implementations
have as a KL target and this one does not.

  reward        mean over the iteration's rollouts
  raw_std       advantage spread BEFORE normalising; near zero means the
                buffer has stopped carrying information
  ratio         mean policy ratio, how far the update moved the policy
  clipped       fraction of samples outside the trust region
  value_loss    how well the critic predicts returns
  cfm           the flow-matching loss the ratio is built from
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(
    0,
    "D:/75128/Desktop/plugrl-work/plugrl-server/.claude/worktrees/"
    "fix-pi0-image-mask-batching/tests",
)

from test_fpo_learns_anything import (  # noqa: E402
    ENVS,
    ITERATIONS,
    POLICIES,
    TARGET,
    _algo,
)
from test_fpo_tree_observations import _tree_env_obs  # noqa: E402
from plugrl_server.common.data_utils import unbatch_aggregate  # noqa: E402
from plugrl_server.policy.state import slice_policy_step_state  # noqa: E402


def iteration_with_metrics(algo, rng):
    prev_node = {env: (-1, "") for env in range(ENVS)}
    terminated = {env: False for env in range(ENVS)}
    obs = _tree_env_obs(ENVS, rng)
    rewards = []
    for round_index in range(400):
        action, runtime_state = algo.infer(obs)
        step_state = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        next_obs = _tree_env_obs(ENVS, rng)
        obs_list = unbatch_aggregate(obs, aggregate_method="concat")
        next_obs_list = unbatch_aggregate(next_obs, aggregate_method="concat")
        done = round_index % 3 == 2
        taken = np.asarray(action, dtype=np.float32).reshape(ENVS, -1)
        for env in range(ENVS):
            reward = -float(((taken[env] - TARGET) ** 2).mean())
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
            break
    algo.pre_learn()
    _, metrics = algo.learn()
    algo.post_learn()
    return float(np.mean(rewards)), metrics


def run(dtype, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    algo = _algo(POLICIES[dtype])
    rng = np.random.default_rng(seed)
    print(f"\n=== {dtype}, seed {seed} ===")
    print(
        f"{'it':>3} {'reward':>9} {'raw_std':>9} {'ratio':>7} {'clipped':>8} "
        f"{'v_loss':>9} {'cfm':>9}"
    )
    for i in range(ITERATIONS):
        reward, m = iteration_with_metrics(algo, rng)
        f = m["fpo"]
        print(
            f"{i + 1:3d} {reward:+9.4f} {f.get('advantages_raw_std', float('nan')):9.4f} "
            f"{f['policy_ratio_mean']:7.4f} {f['clipped_ratio_mean']:8.4f} "
            f"{m['losses']['value_loss']:9.2e} {f['cfm_loss_mean']:9.4f}"
        )


if __name__ == "__main__":
    run("float32", 4)
    run("float32", 1)
