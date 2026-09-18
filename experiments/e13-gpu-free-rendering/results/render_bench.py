"""What does rendering a LIBERO step cost without a GPU?

Run twice by render_bench.sh, once with NVIDIA's EGL and once with Mesa's,
the second with no GPU visible to the process at all. Same task, same
actions, same machine: the ratio is the price of software rendering.
"""

import os
import sys
import time

import numpy as np

from plugrl_env_client.envs.libero.libero_env import LiberoConfig, LiberoEnv

LABEL = os.environ.get("BENCH_LABEL", "unlabelled")
STEPS = int(os.environ.get("BENCH_STEPS", "60"))
ACTION = np.array([[0.0] * 6 + [-1.0]], dtype=np.float32)

t0 = time.time()
env = LiberoEnv(LiberoConfig(task_suite_name="libero_spatial", task_id=0))
build_s = time.time() - t0

t0 = time.time()
env.reset()
reset_s = time.time() - t0

# A few steps first: the first render allocates buffers and compiles shaders.
for _ in range(5):
    env.step(ACTION)

t0 = time.time()
for _ in range(STEPS):
    env.step(ACTION)
step_s = time.time() - t0

print(
    f"BENCH\t{LABEL}\tbuild={build_s:.1f}s\treset={reset_s:.2f}s\t"
    f"steps={STEPS}\ttotal={step_s:.2f}s\tfps={STEPS / step_s:.2f}",
    flush=True,
)
sys.stdout.flush()
os._exit(0)  # skip EGL teardown, which throws on the Mesa path
