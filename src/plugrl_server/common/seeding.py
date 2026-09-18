"""Make `--seed` mean something.

The flag has existed since the first release and only ever reached the
experiment's name. Every generator the server draws from - the denoising noise
in a flow policy, the minibatch indices in a buffer, the random actions in the
dummy policy - started from whatever entropy the process happened to get, so
two runs of the same configuration took different trajectories and no result
could be reproduced exactly.

What seeding does and does not buy:

* **A single env client is reproducible.** Requests arrive in one order, so the
  draws happen in one order.
* **Several clients are not.** The server batches whatever inference requests
  have arrived when the scheduler fires, so the composition of a batch depends
  on arrival timing. The same seed then gives a different sequence of draws.
  Fixing that needs per-request generators, not a global seed.
* **Nothing here is about cuDNN determinism.** Convolution algorithm choice can
  still vary run to run; `torch.use_deterministic_algorithms` is deliberately
  not set, because it turns unsupported kernels into errors and would change
  which policies can run at all.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, on CPU and on every visible GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Subprocesses that read PYTHONHASHSEED at startup, such as a dataloader
    # worker, inherit the same value. It does not affect this interpreter,
    # whose hash seed was fixed before main() ran.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    logger.info(
        f"Seeded python, numpy and torch with {seed}. "
        "One env client is then reproducible; several are not, because batch "
        "composition depends on when their requests arrive."
    )
