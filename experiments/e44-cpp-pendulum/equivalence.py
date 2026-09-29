"""Is pendulum_client.cpp's Pendulum gymnasium's Pendulum-v1?

    python equivalence.py ./pendulum_client [TRIALS]

For each trial: a random start state and 200 random torques, some outside
[-2, 2] so the clip is exercised. gymnasium steps from that state (set on the
unwrapped env after a reset); the C++ binary steps from the same state with
the same float32 torques, in its --check mode, with no network. Prints the
largest differences in observation and reward, and whether the time limit
ended both on the same step.
"""

import subprocess
import sys

import gymnasium as gym
import numpy as np

binary = sys.argv[1]
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(0)
env = gym.make("Pendulum-v1")

worst_obs = worst_reward = 0.0
same_limit = True
for trial in range(trials):
    th0 = float(rng.uniform(-np.pi, np.pi))
    thdot0 = float(rng.uniform(-1.0, 1.0))
    torques = rng.uniform(-2.5, 2.5, 200).astype(np.float32)

    env.reset(seed=trial)
    env.unwrapped.state = np.array([th0, thdot0])
    py = []
    for u in torques:
        obs, reward, terminated, truncated, _ = env.step(
            np.array([u], dtype=np.float32)
        )
        py.append(
            (obs.astype(np.float64), float(reward), bool(terminated or truncated))
        )

    # %.9g round-trips a float32 exactly; repr round-trips a double.
    stdin = "\n".join(f"{float(u):.9g}" for u in torques) + "\n"
    out = subprocess.run(
        [binary, "--check", repr(th0), repr(thdot0)],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    cpp = []
    for line in out:
        if not line.strip():
            continue
        a, b, c, r, t = line.split()
        cpp.append((np.array([float(a), float(b), float(c)]), float(r), t == "1"))

    assert len(cpp) == len(py) == 200, (len(cpp), len(py))
    for (po, pr, pt), (co, cr, ct) in zip(py, cpp):
        worst_obs = max(worst_obs, float(np.abs(po - co).max()))
        worst_reward = max(worst_reward, abs(pr - cr))
        same_limit &= pt == ct
    print(
        f"trial {trial:2d}  start ({th0:+.3f}, {thdot0:+.3f})  "
        f"final obs gym {py[-1][0].round(4)} cpp {cpp[-1][0].round(4)}"
    )

print(f"\n{trials} trials x 200 steps")
print(f"largest observation difference: {worst_obs:.3e}")
print(f"largest reward difference:      {worst_reward:.3e}")
print(f"time limit on the same step:    {'yes' if same_limit else 'NO'}")
