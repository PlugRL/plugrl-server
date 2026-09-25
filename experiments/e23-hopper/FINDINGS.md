# E23: FPO learns Hopper, where episodes end because the agent falls

2026-09-25 · Windows 11, CPU only · `fpo-policy × fpo × Hopper-v5`, three
seeds, 409,600 steps each · protocol: [`PROTOCOL.md`](PROTOCOL.md) ·
amendments: [`AMENDMENT.md`](AMENDMENT.md) - the first attempt was ended by a
machine restart and rerun once; the machine was shared, so P3 is not read

---

## The result

Every learning curve this platform had produced on a CPU was HalfCheetah's,
which never terminates: every episode is cut at 1,000 steps. So one path had
never run inside a learning run - an episode that ends because the agent
failed, reported by the client as `terminated`, recorded by the server as
such, and not bootstrapped past by GAE. Hopper-v5 terminates whenever the
hopper falls, which the untrained policy does within about 21 steps.

**FPO learns it, on all three seeds.**

| seed | start | end | gain | episode length, start → end | best iteration |
| --- | --- | --- | --- | --- | --- |
| 0 | 16.3 | **1587.1** | +1570.8 | 21.0 → 555.1 | 2320.3 |
| 1 | 19.1 | **2005.8** | +1986.7 | 22.4 → 716.6 | 2575.4 |
| 2 | 14.5 | **1776.1** | +1761.6 | 20.4 → 598.4 | 2316.0 |

`start` is the first iteration, `end` the mean of iterations 91 to 100.
**P1 holds**: every seed ran 100 iterations, wrote its checkpoint at 409,600,
has no traceback in its server log and a client that exited 0. **P2 holds,
3 of 3**: every `end` is above 500, and above 1,000, which is past a policy
that has only learned to stand still. The first iteration of every seed
matches the pilot's to the displayed precision (16.25, 19.05, 14.48), because
collection before the first update is deterministic here.

P3, the wall clock, came in at 75.8 minutes against a bound of 150. It is not
read: another project's processes were using the CPU from 16:02 (amendment 2).

---

## What the termination path had to carry

At the start of each run an iteration of 4,096 steps holds about 195 episode
ends, each a `terminated` flag, a reset and a GAE boundary - against about 4
per iteration on HalfCheetah. By the end, episodes last 555 to 717 steps, so
most still end by falling rather than at the 1,000-step cut. The whole run is
a sustained test of the path HalfCheetah never used, and it learned through
it. The code reading in the protocol said terminated steps are not
bootstrapped past; this is the end-to-end measurement that agrees.

---

## What this does and does not support

**Supported:**

* FPO on this platform learns Hopper-v5 from a random initialisation, with
  its defaults tuned for nothing in particular: from about 17 to 1,587-2,006
  in 409,600 steps on three seeds.
* The platform carries a task whose episodes terminate, at about 195 episode
  ends per iteration, through 100 iterations on three concurrent seeds,
  without an error.

**Not supported:**

* Anything comparing FPO with another algorithm on Hopper. There was no
  comparison arm.
* That the truncation path is correct in every case. FPO's
  `treat_truncated_as_done=False` drops each truncated episode's final
  transition rather than bootstrapping it (protocol, "What reading the code
  already says"); a policy that reaches the 1,000-step cut often would lean
  on that path, and these mostly fall first.

---

## Reproducing

```bash
bash run.sh            # three seeds x 409,600 steps, three at a time
python summarise.py    # P1, P3, P2, then the figures above
```

The first attempt's logs, ended by the restart at 20 of 100 iterations, are
in `results/attempt-1/` and are not read.
