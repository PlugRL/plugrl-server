"""What does loading a checkpoint change, measured inside one process?

Three evaluations of the same actor weights returned three different scores:
29 of 50 with no checkpoint loaded, 35 loading a dump of the policy's own
state_dict, 37 loading the lr0 control's. A grouped diff has already shown the
actor tensors are bit-identical between the base and the lr0 checkpoint, so
the weights that generate actions are not what differs.

This constructs the policy exactly as the server does, snapshots everything
that could plausibly matter, loads a checkpoint, and snapshots again:

  weights   every tensor in state_dict, compared exactly
  rng       CPU and CUDA generator states, compared exactly
  mode      how many modules are in training mode

Whatever comes back different is the thing the evaluation is actually keying
on, and whatever comes back identical can stop being suspected.
"""

import pathlib
import sys

import torch

from plugrl_server.common.checkpoint_manager import load_checkpoint_from_path
from plugrl_server.policy.openpi.openpi_policy import Pi0Policy, Pi0PolicyConfig

R = pathlib.Path("/home/gotham/tmp/plugrl")
# No forward pass happens here, only construction and comparison, so this
# runs on the CPU and competes with nothing.
DEVICE = "cpu"


def snapshot(policy):
    return {
        # On CPU: two snapshots of a 3B policy do not fit beside it on one card.
        "weights": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            torch.cuda.get_rng_state().clone()
            if torch.cuda.is_available() and DEVICE != "cpu"
            else None
        ),
        "training": sum(1 for m in policy.modules() if m.training),
        "modules": sum(1 for _ in policy.modules()),
    }


def main(ckpt_dir: str, seed: int = 7):
    from plugrl_server.common.seeding import seed_everything

    seed_everything(seed)
    policy = Pi0Policy(
        Pi0PolicyConfig(
            name="pi05_libero",
            checkpoint_path=R / "ckpt" / "pi05_libero",
            device=DEVICE,
        )
    )
    before = snapshot(policy)
    print(f"modules in training mode before: {before['training']}/{before['modules']}")

    checkpoint = load_checkpoint_from_path(pathlib.Path(ckpt_dir))
    policy.load_state_dict(checkpoint.model)
    after = snapshot(policy)

    changed = []
    for k, v in before["weights"].items():
        w = after["weights"][k]
        if v.shape != w.shape or not torch.equal(v, w):
            changed.append(k)
    print(f"weights changed by the load: {len(changed)} of {len(before['weights'])}")
    for k in changed[:8]:
        print(f"  {k}")

    print(f"cpu rng changed:  {not torch.equal(before['cpu_rng'], after['cpu_rng'])}")
    if before["cuda_rng"] is not None:
        print(
            "cuda rng changed: "
            f"{not torch.equal(before['cuda_rng'], after['cuda_rng'])}"
        )
    print(
        f"modules in training mode after: {after['training']}/{after['modules']}"
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7)
