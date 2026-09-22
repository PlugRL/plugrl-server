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

---

## 3. The machine filled up, so the run waits and no longer names its cards

**2026-09-21, an hour after amendment 2. Still no registered data.**

Amendment 2 said the run had moved to GPUs 5, 6 and 7. That was true for about
forty minutes. At 17:50 an eight-process job took **15,876 MiB on every one of
the eight cards**, leaving roughly 8,700 MiB free on each. The run's model card
alone needs 24,095 MiB. There was nowhere on the machine to put it, and the
high indices were worth nothing, because the neighbour was on all of them.

The protocol had already written the rule:

> If the cluster's GPUs are occupied by other work, the run waits rather than
> squeezing beside it.

So it waits. The orchestrator was replaced before it could launch into a full
machine: it no longer names cards in advance. It polls until three cards are
each under 500 MiB in use, takes the three highest of those, and has **no
timeout and no fallback that starts anyway** - because a fallback that starts
anyway is the thing the rule forbids.

What amendment 2 got right is the reason for preferring high indices, and that
still applies when the choice exists. What it got wrong was naming them: a
fixed choice made at 17:42 was already wrong at 17:50. The cards are now chosen
at the moment the run starts, and the choice is logged.

### `eval-baseline` ran beside that job, and its row is kept

The cell started at 17:39, when GPU 0 was empty. The eight-process job arrived
at 17:50, while it was running. It finished at 18:50 with **29 of 50**, exit
status 0, and the post-processor reconciled the server's episode record with
the client's, so the row is marked valid.

It took 4,101 s where E11's equivalent took 1,937 s - 2.1x, from sharing the
card. That is a fact about the clock, not about the policy: the cell replays 50
fixed initial states under a fixed seed, and contention changes neither the
actions nor the environment's dynamics. Prediction 6, the only one about wall
clock, is about training.

This is recorded now, before the remaining cells are run and before it is known
whether the number helps or hurts. The alternative reading - that the cell
squeezed beside other work and should be discarded - is available to a reader,
and the evidence for it is here rather than left out.

For the record, E11 measured **26 of 50** on this task. The protocol required
re-measuring rather than reusing that number, because E11's evaluations predate
the seeding fix. The two intervals overlap heavily, so this is not a
contradiction; it is the reason the protocol asked.

### One more thing the move would have broken

The memory recorder samples `nvidia-smi -i 0` and `-i 1` from outside the
server, because nothing inside records it and **which card holds what is the
point**. `CUDA_VISIBLE_DEVICES` does not remap that. Had the run moved to cards
5 and 6 with the recorder untouched, `results/train.tsv`'s peak memory per card
- the evidence prediction 1 turns on - would have been two idle cards. The
recorder now follows the cards the run is given.

---

## 4. The run reached step 40,960 and lost the checkpoint, and iteration 9 stands in for iteration 10

**2026-09-22, after the run ended. This is the first amendment written with
registered data in hand, and it changes no registered value.**

### What happened

The run ended at 07:08:22 on 2026-09-22 after 43,206 s. Not from memory, and
not from any neighbour:

```
[07:08:22] clients exit 124 after 43206s (server_died=0)
[07:08:55] teardown clean
           === out of memory? ===
           0
           === peak memory, GPU0 and GPU1 ===
           gpu0_peak_mib=20919  gpu1_peak_mib=9925
```

Exit status 124 is `timeout`. The env clients run under `timeout 43200` in
`e14_train.sh` - a **twelve-hour budget in our own harness**. The run needed
about thirteen. The server was alive throughout and shut down cleanly.

### The tenth iteration computed and its checkpoint did not survive

Step 40,960 is exactly ten iterations of the 4,096-transition buffer, and the
run got there: the tenth learn step finished and the server was writing
`tmp_40960/` when the clients were killed. That directory holds a single file,
short of its own declared length:

```
4100        COMPLETE    expected=8528818570  actual=8528818570  missing=0
36870       COMPLETE    expected=8528818570  actual=8528818570  missing=0
tmp_40960   TRUNCATED   expected=8528818570  actual=8355110912  missing=173707658
```

The expected size is computed from the safetensors header's own declared
tensor offsets, not guessed from the other files. `config.yaml`, `metadata.pt`
and `optimizer.pt` were never written at all.

**So `eval-iter10` has nothing to evaluate.** Iteration 9 at step 36,870 is the
last complete checkpoint, and the evaluation cell is labelled **`iter09`**, not
`iter10`, so that no row in `results/eval.tsv` claims to be a checkpoint that
does not exist. `eval-iter10-repeat` becomes `eval-iter09-repeat` and still
does its job, which was never about the policy: it tests whether the seeding
fix makes an evaluation reproducible.

The alternative - resuming from step 36,870 to produce a tenth checkpoint -
was rejected. The cause here is internal, our own budget, so the protocol's
one restart for an external cause does not cover it and is spent besides. A
resumed run would also be a different run: fresh clients, different batch
composition by arrival. Nine iterations documented beats ten assembled from
two runs.

### What the run did establish

**No `OutOfMemoryError`, across all ten iterations.** Per iteration, from
`results/train.tsv` - wall clock checkpoint to checkpoint, with iteration 1
measured from `server listening`, and peak memory per card:

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| wall s | 4696 | 4629 | 4680 | 4603 | 4639 | 4642 | 4636 | 4645 | 4660 |
| card 1 MiB | 18504 | 19946 | **20919** | 20086 | 19966 | 20066 | 20046 | 19986 | 19986 |
| card 2 MiB | 9071 | 9071 | **9925** | 9071 | 9071 | 9071 | 9071 | 9071 | 9071 |

Every wall clock is under prediction 6's bound of 4,862 s, and the spread
across nine iterations is 93 s - 2%. **Prediction 6 holds.**

On memory, prediction 1's stated worry was that "ten allocate no more state,
but the allocator may still grow". It grew for three iterations, touched
20,919 MiB in iteration 3, and then **came back down** to about 20,000 and
stayed within 120 MiB of that for the remaining six. The high-water mark of
20,919 is a single iteration's transient, not the level the run sits at, and
quoting it as "the peak" without that sentence would overstate how close to
the card this configuration actually runs.

That table bears on a question this amendment series raised, without settling
it. Iteration 2 ran while the neighbouring jobs on the other cards were down
and iteration 3 while all three were up, and the difference is 51 s, inside
the natural spread; the machine ran at a load average near 193 on 128 cores
for much of the night, including a root process at roughly 61 cores.

**That is one pair of iterations, not a controlled test.** The comparison
rests on two snapshots of the GPU process table taken an hour apart, and the
load timeline that would turn it into evidence does not exist - see the second
failure recorded below. What can be said without it is narrower and still
worth saying: across nine iterations spanning a whole night on a shared
machine, the spread is 93 s, so whatever the load was doing, this run's wall
clock did not follow it.

### Two failures of ours, recorded

**The budget was checkable before the run started, and was not checked.** The
two-iteration probe took 9,044 s, which is 4,522 s per iteration; ten
iterations plus startup was therefore always going to need about 45,600 s
against a 43,200 s budget. That arithmetic needed the probe's own number and
nothing else. It would not have changed what was allowed - restarting for an
internal cause is not permitted and the allowance was spent - but it would
have turned a discovery into a plan, and the substitution of iteration 9 would
have been declared in advance instead of after the fact.

**The load timeline for this run does not exist.** Amendment 3 promised a
recorder sampling load average and per-process CPU on a fixed interval so that
attribution would have something under it. The tunnel to the cluster dropped
at about 22:50 and did not return until the afternoon; the watcher installed
the recorder on reconnect, by which time the run was over. The timings above
survive because they come from checkpoint mtimes on disk. The CPU timeline
cannot be reconstructed, and no claim in the findings will rest on one.

---

## 5. Prediction 5 is about to pass without testing anything, so a second baseline was added

**2026-09-22 19:05, written before the added cell has run and before the
registered repeat has returned.**

### The registered test has gone vacuous

Prediction 5 reads:

> **`eval-iter10-repeat` returns exactly the same successes as `eval-iter10`.**
> Falsified by any difference. This is the seeding fix or it is not.

Its purpose is stated in the protocol: the repeat "is not about the policy",
it checks the seeding fix under real conditions. E12 found that `--seed`
reached only the run's name, and the fix in #31 is what makes a single-client
evaluation reproducible at all.

The trained policy scores **0 of 50** at iterations 1, 2, 5 and 9. So the
repeat will score 0 of 50, and the prediction will hold. But a policy that
fails every episode agrees with itself under any seed, under a different seed,
or under no seeding at all. **Nothing about reproducibility can be inferred
from two zeros.** The prediction as registered will pass and will have tested
nothing, and reporting it as a confirmation of the seeding fix would be
claiming evidence that does not exist.

### What was added

`eval-baseline-repeat`: the unmodified checkpoint, a second time, same seed,
same 50 initial states in the same order. The baseline's 29 of 50 is the only
non-degenerate number this experiment has, and repeating it is the test the
registered repeat was meant to be.

It is **additive**. It changes no registered value, discards no measurement,
and replaces nothing - `eval-iter09-repeat` still runs, and is still reported
with the note that it is vacuous. The cost is about 45 minutes on cards that
are otherwise idle.

Whether it returns 29 is not known as this is written. If it does, the seeding
fix does what E12's finding said it must. If it returns anything else, that is
a real result and a more important one than anything else in this experiment,
because it would mean the single-client evaluations this project has been
reporting - including E11's 26 of 50 and this run's own baseline - are not
reproducible after all.

### The general point, since this is the third time

A pre-registered prediction can be satisfied by a degenerate outcome. Writing
predictions before the data is what keeps them honest; it does not guarantee
they remain informative once the data arrives. This one was written expecting
a policy with a non-zero success rate at iteration 10, and the collapse to
zero took its content away. The protocol could not have known that - it
predicted the collapse in the same document - but the findings have to say
which predictions were tested and which merely passed.
