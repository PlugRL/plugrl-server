"""Write the freshly-constructed policy's state_dict out in PlugRL's own keys.

Every checkpoint comparison so far has been blocked by the same thing: the
base openpi file and a PlugRL checkpoint share no tensor names, so "how far
did one iteration move the weights from the base" could not be measured. A
policy built exactly as the server builds it, dumped before anything trains,
is the base in the right keys.

It is also the artefact needed to explain the lr0 control. That run trained
one iteration at learning_rate 0, where no weight can change, and its
checkpoint evaluated 37 of 50 where the base evaluates 29 - twice, on the
same states, from a harness shown to be deterministic. Either the actor did
change, which this measures, or it did not and something outside the weights
accounts for the difference.
"""

import pathlib
import sys

import torch
from safetensors.torch import save_file

from plugrl_server.policy.openpi.openpi_policy import Pi0Policy, Pi0PolicyConfig

R = pathlib.Path("/home/gotham/tmp/plugrl")


def main(out_dir: str, device: str):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = Pi0PolicyConfig(
        name="pi05_libero",
        checkpoint_path=R / "ckpt" / "pi05_libero",
        device=device,
    )
    policy = Pi0Policy(config)
    state = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    print(f"tensors: {len(state)}")
    save_file(state, str(out / "model.safetensors"))
    print(f"wrote {out / 'model.safetensors'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cuda")
