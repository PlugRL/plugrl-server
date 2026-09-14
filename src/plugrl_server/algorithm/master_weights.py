"""Float32 master copies for half-precision parameters."""

from __future__ import annotations

from collections.abc import Iterable

import torch

_HALF_PRECISION = (torch.float16, torch.bfloat16)


class MasterWeights:
    """Lets an optimizer take steps that a half-precision parameter cannot hold.

    The spacing between neighbouring bfloat16 values is about 0.8% of the value.
    An optimizer step smaller than that rounds away, and without a copy in
    higher precision such steps never add up. Pi0Policy's action expert is held
    in bfloat16, and at a learning rate of 1e-5 most of its linear weights did
    not move at all.

    Every trainable float16 or bfloat16 parameter gets a float32 copy, and the
    optimizer steps the copy in its place:

        optimizer.zero_grad(); master.clear_model_grads()
        loss.backward()
        master.grads_to_masters()
        optimizer.step()
        master.masters_to_model()

    A step too small for the model is kept in the copy and reaches the model
    once the steps add up to a representable change.

    Float32 parameters go to the optimizer as they are, and frozen parameters are
    left out, so a model with no half-precision parameters gives the optimizer
    exactly the parameters it would otherwise get.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        self.pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        self.optimizer_params: list[torch.nn.Parameter] = []
        for param in params:
            if not param.requires_grad:
                continue
            if param.dtype in _HALF_PRECISION:
                master = torch.nn.Parameter(param.detach().to(torch.float32).clone())
                self.pairs.append((param, master))
                self.optimizer_params.append(master)
            else:
                self.optimizer_params.append(param)

    def clear_model_grads(self) -> None:
        for param, _ in self.pairs:
            param.grad = None

    def grads_to_masters(self) -> None:
        """Move each half-precision gradient onto its copy, in float32."""
        for param, master in self.pairs:
            if param.grad is None:
                master.grad = None
                continue
            master.grad = param.grad.detach().to(torch.float32)
            param.grad = None

    @torch.no_grad()
    def masters_to_model(self) -> None:
        """Round each copy back into the model."""
        for param, master in self.pairs:
            param.copy_(master)

    @torch.no_grad()
    def sync_from_model(self) -> None:
        """Reset each copy from the model, as after loading its weights.

        A copy's progress below the model's precision is not in the model's
        weights, so it does not survive a checkpoint round trip.
        """
        for param, master in self.pairs:
            master.copy_(param)
