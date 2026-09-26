"""How often does DPPO's logprob clamp [-5, 2] bind, per denoising step?

Builds each policy fresh (random initialisation, as every E28 cell starts),
samples a batch of actions with the Hopper variant's sampling_noise_level,
and counts the per-element logprobs the loss would clamp. A clamped element
contributes no gradient and a ratio of exactly 1.
"""

import sys

import numpy as np
import torch

from plugrl_server.policy.dppo.dppo_policy import DPPOPolicy, DPPOPolicyConfigHopper
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig

LEVEL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
B = 4096
torch.manual_seed(0)
obs = {
    "states": {
        "obs": np.random.default_rng(0).standard_normal((B, 11)).astype(np.float32)
    }
}

policies = {
    "fpo-policy": FPOPolicy(FPOPolicyConfig(obs_dim=11, action_dim=3, device="cpu")),
    "dppo-policy": DPPOPolicy(DPPOPolicyConfigHopper(device="cpu")),
}
print(f"sampling_noise_level {LEVEL}; fraction of logprob elements outside [-5, 2]")
for name, policy in policies.items():
    with torch.no_grad():
        _, state = policy.get_action_and_runtime_state(obs, sampling_noise_level=LEVEL)
    lp = state.logprob  # [B, steps, horizon, dim]
    high = (lp > 2).float().mean(dim=(0, 2, 3))
    low = (lp < -5).float().mean(dim=(0, 2, 3))
    t = state.obs.t[0]
    print(
        f"{name}: {lp.shape[1]} steps, overall above 2: {float((lp > 2).float().mean()):.3f}, "
        f"below -5: {float((lp < -5).float().mean()):.4f}, mean entropy {float(state.entropy.mean()):.3f}"
    )
    for i in range(lp.shape[1]):
        print(
            f"    step {i:2d} t={float(t[i]):6.3f}  above 2: {float(high[i]):.3f}  below -5: {float(low[i]):.4f}"
        )
