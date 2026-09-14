# E11 amendments

`PROTOCOL.md` allows at most two adjustments to Stage C's starting values. Each is recorded here with the observation that forced it, before the run that uses it. A result that needed a third adjustment is reported as not obtained.

---

## Adjustment 1 - `batch_size` 32 to 8

**Observation, 2026-09-14.** A Stage C harness check ran the protocol's starting values on the chosen task, libero_10 task 8, with a 64-transition buffer. It was not a protocol run.

The server ran out of GPU memory in the first learn, before any gradient step:

```
pre_learn -> FPOBuffer.prepare_fpo_fields -> compute_cfm_loss
  -> Pi0Policy._predict_v -> denoise_step -> gemma_expert self-attention
torch.OutOfMemoryError: Tried to allocate 490.00 MiB. GPU 0 has a total
capacity of 23.56 GiB of which 436.81 MiB is free. 21.48 GiB is allocated by
PyTorch, and 1.33 GiB is reserved by PyTorch but unallocated.
```

GPU 0 held 7.5 GB through collection and reached 23.7 GB at the first learn. The memory was allocated, not fragmented, so an allocator setting would not recover it.

**Why the batch.** FPO evaluates the action expert on `batch_size * n_samples_per_action` action sequences at once: 32 * 4 = 128. Each sequence carries the whole observation prefix, about 968 tokens, as key-value cache repeated across the attention heads. The allocation that failed, 490 MiB, is one such key tensor for 128 sequences: 128 * 8 heads * 978 tokens * 256 * 2 bytes.

The learn step's own forward and backward run at the same 128, so it would fail as well.

**Adjustment.** `batch_size` goes from 32 to 8, which brings the expert's batch from 128 to 32. `n_samples_per_action` stays at 4: the protocol already halved it from the default to save memory, and FPO's likelihood-ratio estimate averages over those samples.

**What it changes.** Each learn iteration takes 4096 / 8 * 4 = 2048 optimizer steps instead of 512, each on a quarter of the data.

**Checked before use.** A second harness check ran with `batch_size` 8 on the same task and buffer, using server code identical file for file to plugrl-server `2e73693`. It completed a full learn and wrote a checkpoint.

GPU 0 peaked at 19,208 MiB, sampled every 15 s. PyTorch's caching allocator does not return memory within a run, so that reading bounds the peak.

The fix for bfloat16 rounding (`NOTES.md`, 2026-09-14) adds float32 copies of the trainable half-precision weights. A check at `batch_size` 8 with that fix peaked at 20,440 MiB, and that is the configuration Stage C runs.
