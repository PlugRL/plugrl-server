"""A per-sample scalar in the observation must batch like any other leaf.

pi0's observation carries one boolean per camera, `image_mask`, which the
buffer stores as a `(capacity,)` array. DPPO's learn reads the buffer through
a torch DataLoader, one index at a time, and indexing a 1-D array with an int
gives a numpy scalar (`np.bool_`), not an array. `stack_numpy_tree` treats
anything that is not an `np.ndarray` as a mapping, so the first learn step
raised `AttributeError: 'numpy.bool_' object has no attribute 'keys'` - the
first time DPPO trained pi0 (E25's pilot). FPO never met it: it reads the
buffer by slice, and fpo-policy's observations have no per-sample scalars.
"""

from __future__ import annotations

import numpy as np
import torch

from plugrl_server.buffer.rollout_buffer import GAEBuffer

FRAMES = 4


def _obs(k: int) -> dict:
    # Shaped as a policy hands a train state to the buffer: a leading batch
    # dimension of 1 on every leaf, so image_mask's per-sample shape is ().
    return dict(
        image_mask=dict(
            base=np.array([k % 2 == 0]),
            wrist=np.array([True]),
        ),
        state=np.full((1, 3), float(k), dtype=np.float32),
    )


def _train_state(k: int) -> dict:
    return dict(
        obs=_obs(k),
        action=np.zeros((1, 2), dtype=np.float32),
        logprob=np.zeros((1, 2), dtype=np.float32),
        value=np.zeros((1,), dtype=np.float32),
    )


def _filled_buffer() -> GAEBuffer:
    """FRAMES one-frame episodes, each terminated, so nothing needs a bootstrap."""
    buffer = GAEBuffer(8, _train_state(0))
    for k in range(FRAMES):
        buffer.add_frame(
            prev_node=(-1, buffer.buffer_signature),
            train_state=_train_state(k),
            reward=1.0,
            terminated=True,
            truncated=False,
            last_value=None,
            next_terminated=False,
            next_truncated=False,
        )
    buffer.compute_advantages_and_returns()
    return buffer


def test_one_item_keeps_every_leaf_an_array():
    buffer = _filled_buffer()

    item = buffer.train_state_storage.get_item(1)

    mask = item["image_mask"]["base"]
    assert isinstance(mask, np.ndarray), type(mask)
    assert mask.shape == ()
    assert not bool(mask)  # frame 1: k % 2 == 0 is False
    assert isinstance(item["state"], np.ndarray)
    assert item["state"].shape == (3,)


def test_dppo_s_dataloader_batches_a_per_sample_scalar():
    """DPPO's own way of reading the buffer: a DataLoader over it, by index."""
    buffer = _filled_buffer()
    loader = torch.utils.data.DataLoader(
        buffer, batch_size=FRAMES, shuffle=False, collate_fn=buffer.collate_fn
    )

    obs, action, *_ = next(iter(loader))

    assert obs["image_mask"]["base"].dtype == torch.bool
    assert obs["image_mask"]["base"].shape == (FRAMES,)
    assert torch.equal(
        obs["image_mask"]["base"], torch.tensor([True, False, True, False])
    )
    assert torch.equal(obs["image_mask"]["wrist"], torch.ones(FRAMES, dtype=torch.bool))
    assert obs["state"].shape == (FRAMES, 3)
    assert action.shape == (FRAMES, 2)


def test_a_slice_is_unchanged():
    """FPO's way - by slice - already returned arrays and must still."""
    buffer = _filled_buffer()

    item = buffer.train_state_storage.get_item(slice(0, FRAMES))

    assert item["image_mask"]["base"].shape == (FRAMES,)
    assert item["state"].shape == (FRAMES, 3)
