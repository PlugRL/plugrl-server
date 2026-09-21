# E14 amendments

Changes made after [`PROTOCOL.md`](PROTOCOL.md) was written, each dated and
with its reason. The protocol itself is not edited.

---

## 1. The server code on the cluster is now `main`, which deploys #20

**2026-09-21. No registered data had been collected when this was decided.**

### What happened

The first launch of the ten-iteration run died 165 seconds in, before a single
transition was collected:

```
TypeError: WebSocketAgentServer.__init__() got an unexpected keyword
argument 'feedback_wait_timeout'
```

The protocol asked for `--seed 7` to mean something, which needs the seeding
fix, so `cli.py` and `common/seeding.py` were copied to the cluster. The
cluster's checkout is not a git clone - it is a source snapshot from E11's
era, which E11's own findings record as `0c36c0e` in 63 of its 65 files. The
new `cli.py` passes `feedback_wait_timeout` to a server that predates the
argument. Copying single files into a snapshot made an incoherent mix, and
nothing checked that it could still start.

### What was done

The whole of `src/plugrl_server` was synced from `main` at **32bc21a**, so the
code on the cluster is one coherent version rather than a snapshot with
patches on top.

### The consequence, which the protocol did not anticipate

That sync **deploys #20**, and E11 did not have it. E11's findings say so
plainly: #20 "was merged but never deployed, so the connection behaviour
described below is the old one" - four feedback timeouts per training run and
420 client reconnect attempts in its second attempt, each discarding the
feedback the client was holding.

So E14 differs from E11 in one more way than the protocol listed. It is the
better version - clients are no longer mistaken for dead while a learn step
runs - but it is a difference, and any comparison of E14's training dynamics
with E11's has to carry it.

This is written before the re-launch rather than discovered in the findings.

### Also changed, for the same reason

A smoke test now runs before the real one: the server's construction is
checked in a throwaway process first. The sixteen hours lost here were not
the failure's fault. The failure took 165 seconds; not noticing took the rest.

---

## 2. The first ten-iteration attempt died of another job's memory

**2026-09-21. No registered data had been collected when this was decided.**

### What happened

The run launched at 17:22:03, reached `server listening` at 17:24:53, and
served rollouts until 17:37:10, when the policy's forward pass raised:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 124.00 MiB.
GPU 0 has a total capacity of 23.56 GiB of which 105.19 MiB is free.
... this process has 16.87 GiB memory in use. Process 60000 has 3.82 GiB
memory in use.
```

`Process 60000` is not ours. Another user's job took 3.82 GiB of GPU 0 while
the run was training - and the two-iteration probe had already measured this
run's own peak at **24,095 MiB of the card's 24,576**, which is 98%. There was
no room for a neighbour of any size. No checkpoint had been written yet; the
first was not due for another hour. The logs are kept at
`e14/evidence/oom-20260921-1737-*`.

### Why this is not prediction 1 being falsified

Prediction 1 is falsified by "any `OutOfMemoryError`", and one occurred, so the
distinction has to be argued rather than assumed.

The prediction asks whether ten iterations fit in the configuration the
protocol declares: one card for the model, one for the master copies. That was
not tested, because the card the prediction is about was not the card the run
had - 3.82 GiB of it belonged to someone else. To report this as a
falsification would be to report another user's scheduling as a fact about
FPO's memory.

The protocol wrote the rule before the data:

> One restart is allowed if the cause is external - another job taking the
> card, the machine rebooting - and that restart is recorded with its reason.

This is that restart, and this is its reason. **The allowance is now spent.** A
second `OutOfMemoryError` falsifies prediction 1 whatever its cause, because a
configuration that cannot survive the machine it runs on has not fit.

### What changed, and what did not

The run moved from GPUs 0, 1 and 2 to **5, 6 and 7**. Nothing else changed. The
low indices are the ones a job with a default `cuda:0` takes first, which is
the likeliest way the neighbour arrived; the high ones are taken last.
`CUDA_VISIBLE_DEVICES` remaps indices, so the server still places its model on
`cuda:0` and the master copies on `cuda:1` - **`master_weights_device=cuda:1`
is unchanged**, as is every other registered value. `e14_train.sh` gained two
optional variables, `SRV_GPUS` and `CLI_GPU`, defaulting to the old indices;
the diff is three lines and the original is kept beside it.

This lowers the chance of a neighbour. It does not remove it. A run that needs
98% of a card, for twelve hours, on a machine it does not own, is exposed the
whole time, and saying so is worth more than the mitigation.

### Also: the guard was wrong a second time, in the opposite direction

Amendment 1 recorded a watcher that looked only for a completion marker, which
failure paths never print. Its replacement looked for `E14_FEASIBILITY_DONE` -
which the runner prints on its way out **whether the run worked or not**. So
the evaluation queue read "training finished", found zero checkpoints, and
started anyway.

Twice the marker's meaning was taken from its name instead of from the script
that prints it. The guard no longer reads markers: it counts checkpoint
directories and requires ten.

The consequence was harmless, and is kept rather than discarded. What the queue
started was `eval-baseline`, which evaluates the unmodified checkpoint and
needs no trained one; the protocol requires that cell regardless. It ran on
GPUs 0 and 1, which the training had just vacated.
