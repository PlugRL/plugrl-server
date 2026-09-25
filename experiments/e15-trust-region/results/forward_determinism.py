"""Is one forward pass of pi0.5 deterministic, before and after loading a checkpoint?

E14 and E15 measured the untrained pi0.5 three times without a checkpoint at
29, 29, 29 of 50, and three times through the checkpoint path - loading a dump
of its own state_dict, bit-identical weights - at 35, 28, 29. A CPU probe then
showed loading changes no actor tensor and no CPU random state. It never
looked at the CUDA random state, because it ran on the CPU.

This runs the part of an evaluation that is not the environment: the prefix
forward through the VLM and the ten denoising steps, on a GPU, with a fixed
input and a fixed seed, twice before loading and twice after. Run it twice -
once as is, once with FORCE_DETERMINISTIC=1 and CUBLAS_WORKSPACE_CONFIG set -
and the answers separate the candidates:

  identical everywhere     a single forward is deterministic; the spread is
                           in the episodes, not the kernels
  differs only after load  loading changes which kernels run
  differs everywhere,      the kernels are non-deterministic, and the warning
  fixed by the flag        list below names them
"""

import os
import pathlib
import warnings

import numpy as np
import torch

from plugrl_server.common.checkpoint_manager import load_checkpoint_from_path
from plugrl_server.common.seeding import seed_everything
from plugrl_server.policy.openpi.openpi_policy import Pi0Policy, Pi0PolicyConfig

R = pathlib.Path("/home/gotham/tmp/plugrl")
CKPT = R / "e14" / "base-statedict"
DETERMINISTIC = os.environ.get("FORCE_DETERMINISTIC") == "1"
if DETERMINISTIC:
    # warn_only: an op with no deterministic implementation warns instead of
    # raising, and the warnings are the list of culprits.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_cond(policy, batch: int = 2, seed: int = 123) -> dict:
    """A model-space observation with every path that matters switched on.

    `fake_diffusion_cond` masks every image out, which would leave the vision
    encoder's output unused. LIBERO sends a base and a wrist camera, so those
    two are real and unmasked here.
    """
    rng = np.random.default_rng(seed)

    def image():
        return torch.from_numpy(
            rng.integers(0, 256, size=(batch, 224, 224, 3), dtype=np.uint8)
        )

    tokens = torch.from_numpy(
        rng.integers(0, 1000, size=(batch, policy.max_token_len))
    ).long()
    token_mask = torch.zeros(batch, policy.max_token_len, dtype=torch.bool)
    token_mask[:, :24] = True
    return dict(
        image=dict(
            base_0_rgb=image(),
            left_wrist_0_rgb=image(),
            right_wrist_0_rgb=torch.zeros(batch, 224, 224, 3, dtype=torch.uint8),
        ),
        image_mask=dict(
            base_0_rgb=torch.ones(batch, dtype=torch.bool),
            left_wrist_0_rgb=torch.ones(batch, dtype=torch.bool),
            right_wrist_0_rgb=torch.zeros(batch, dtype=torch.bool),
        ),
        state=torch.from_numpy(
            rng.standard_normal((batch, policy.action_dim)).astype(np.float32)
        ),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
    )


@torch.no_grad()
def forward(policy, cond: dict, seed: int) -> torch.Tensor:
    """The prefix forward and the denoising loop, as an evaluation runs them."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cache = policy.build_obs_cache(cond)
    batch = cond["state"].shape[0]
    x = policy._initialize_x(batch)
    for t in policy._get_timesteps():
        x, _, _ = policy._denoising_step(x, t.repeat(batch), cond, cond_cache=cache)
        x = policy._iterative_process_action(x)
    return x.float().cpu()


def compare(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs().max().item()
    print(f"  {tag:34s} identical={torch.equal(a, b)!s:5}  max|diff|={diff:.3e}")


def main() -> None:
    print(f"deterministic algorithms forced: {DETERMINISTIC}")
    print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    seed_everything(7)
    policy = Pi0Policy(
        Pi0PolicyConfig(
            name="pi05_libero",
            checkpoint_path=R / "ckpt" / "pi05_libero",
            device="cuda",
        )
    )
    cond = make_cond(policy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        a1 = forward(policy, cond, 11)
        a2 = forward(policy, cond, 11)
        print("no checkpoint loaded, same seed twice:")
        compare("final action", a1, a2)

        before = torch.cuda.get_rng_state().clone()
        checkpoint = load_checkpoint_from_path(CKPT)
        policy.load_state_dict(checkpoint.model)
        after = torch.cuda.get_rng_state().clone()
        print(
            f"CUDA random state changed by the load: {not torch.equal(before, after)}"
        )

        b1 = forward(policy, cond, 11)
        b2 = forward(policy, cond, 11)
        print("after loading, same seed twice:")
        compare("final action", b1, b2)
        print("before against after loading, same seed:")
        compare("final action", a1, b1)

    culprits = sorted(
        {
            str(w.message).split(" does not have a deterministic")[0]
            for w in caught
            if "deterministic" in str(w.message)
        }
    )
    print(f"non-deterministic ops the run used: {len(culprits)}")
    for op in culprits[:20]:
        print(f"  {op[:160]}")


if __name__ == "__main__":
    main()
