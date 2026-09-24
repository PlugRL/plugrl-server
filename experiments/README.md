# Experiments

PlugRL splits reinforcement learning training from the environments it learns
from, into two processes joined by a WebSocket protocol. That is a design
choice, and design choices are usually defended with assertions. These ask
instead:

1. **Is the premise true?** The split was justified by a claim that the two
   dependency stacks cannot coexist. → **E1: no, they can.**
2. **What does it cost?** Every step pays for serialization and a round trip.
   → **E5: 0.5-0.7 ms on loopback**, and the shipped scheduler interval was
   costing more than the boundary itself. → **E7: +0.52 ms more once the
   bytes leave the machine** - and nothing measurable for the network stack
   itself, which is not where the cost turned out to be.
3. **Is it actually portable?** A protocol decouples nothing unless something
   other than this codebase can speak it. → **E2: yes** - two clients written
   against the specification, one of them C++ with no third-party libraries.
   They share an author with the specification, so this shows it is
   sufficient, not that it is clear.
4. **Does any of it train?** → **E6: yes** - FPO on HalfCheetah-v5, three
   seeds.
5. **Does it survive being used?** A clean-machine install found a silent
   corruption of the training buffer across a reconnect. → **E8: fixed** -
   and the first explanation of it was wrong, which the same experiment
   records.
6. **Does it train something that matters?** → **E11 and E14: it trains a
   3B VLA end to end, and the training makes the policy worse.** 29 of 50
   down to 0 of 50 in a single iteration. Ten hypotheses about why have been
   refuted. That is an open defect, not a finding about the algorithm, and it
   is written up as one.

The first of those six answers is negative, and it is the one the project
was built on. The fifth cost a bug and a retracted diagnosis, both of which
are written down. The sixth is unresolved and is the work in progress.

| | Question | Answer |
|---|---|---|
| [`e1-dependency-conflict`](e1-dependency-conflict/) | Can a training stack and an environment stack share one Python environment? | **Yes** - the claim that they cannot is disproved |
| [`e2-cross-language`](e2-cross-language/) | Can the protocol be spoken by something that is not this codebase? | **Yes** - an 843-line C++ client with no third-party libraries drives a real server |
| [`e3-integration-cost`](e3-integration-cost/) | What does adding a new environment or policy cost, here versus elsewhere? | **Never run.** The metrics are pre-registered; the plan is kept, the result does not exist |
| [`e5-boundary-cost`](e5-boundary-cost/) | What does crossing the process boundary cost per step? | **Sub-millisecond** on loopback - a lower bound, not a cross-machine number |
| [`e6-first-learning-curve`](e6-first-learning-curve/) | Does anything here actually learn? | **Yes** - three seeds, episode return from about -300 into the thousands |
| [`e7-cross-machine`](e7-cross-machine/) | What does the boundary cost once packets leave loopback? | **+0.52 ms** on a 184 KiB observation - and the cost is in leaving the machine, not in the network stack |
| [`e8-keepalive-hypothesis`](e8-keepalive-hypothesis/) | Does a long learn step kill the WebSocket connection? | **No** - learns of 190 s, nine times the ping timeout, close nothing. The hypothesis was mine and the measurement refuted it |
| [`e10-vla-forward-cost`](e10-vla-forward-cost/) | What does a VLA forward actually cost, and is the boundary therefore cheap? | **Yes** - 34.9 ms for the policy PlugRL ships and 100.0 ms full size, through openpi's compiled inference, against 1.3 ms to cross a machine: 1.3-3.6% of a step. PlugRL's own uncompiled path is not timed there |
| [`e9-many-clients`](e9-many-clients/) | Does one server stay correct while 8 env clients feed it? | **Yes** - 12 runs at 1/2/4/8 clients, every one exact to the unit, zero warnings. Throughput still rising at 8, which falsified its own prediction |
| [`e11-vla-rl-libero`](e11-vla-rl-libero/) | Can a real VLA be trained through this boundary, and does it help? | **Trained, not helped** - pi0.5 ran end to end with the server's record exact against the clients'; one FPO iteration took the chosen task from 26/50 to 0/50, and the run stopped at 1 iteration of 10 because a second learn step does not fit beside the optimizer state the first one allocated |
| [`e12-cuda-free-rollout`](e12-cuda-free-rollout/) | Does a robomimic-class rollout machine really need CUDA, as E1 concluded? | **No** - that was a packaging default. The CPU build of the same torch takes robomimic's env client from 7.2G with 16 nvidia wheels to 2.7G with none, and LIBERO from 7.8G to 3.4G, sending byte-identical observations. It also found that no run here was reproducible, the policy's noise being unseeded, which is now fixed |
| [`e13-gpu-free-rendering`](e13-gpu-free-rendering/) | And without a GPU to render on? | **Yes, at 1.91x** - ten clients rendering entirely on the CPU, 30 of 30 episodes successful. Its own prediction that the queue would hide the cost, keeping it under 1.3x, is falsified |
| [`e14-ten-iterations`](e14-ten-iterations/) | Over ten iterations, what does FPO do to a full-size pi0.5? | **It collapses in one, and the cause is not located.** Nine iterations ran with no OOM, their wall clocks spread over 93 s; the policy went from 29 of 50 to 0 of 50 across the first update and never recovered. Five candidate explanations were tested and refuted, and the document has been corrected twice |
| [`e15-trust-region`](e15-trust-region/) | Is that collapse a function of how far one iteration moves the policy? | **No.** A five-point `clipping_epsilon` sweep is not monotone. Under 1% of relative movement in pi0.5's action expert takes it from 29 of 50 to 0 of 50, and halving the movement does not halve the damage - four independent ways of moving less all score zero. Its toy control shows FPO learns a bandit and does not hold it |

Each directory has a `FINDINGS.md` stating what was asked, what came back,
and what it does and does not support. E1 is the only experiment that
installs from upstream forks, and its scripts pin those forks' SHAs at the
top, as resolved on the day they ran, so a rerun measures the same thing
rather than whatever the forks have become. The rest fix less: E2 installs
the two local checkouts plus PyPI version ranges, and E5-E8 run against a
`.venv` that already exists, so rerunning those reproduces a working tree and
a range, not an exact stack.

## E1 disproved the hypothesis it was written to test

The starting claim was that a modern training stack (torch 2.7, CUDA) and an
environment stack (`cython<3`, `mujoco-py`, `robosuite<1.5`) cannot coexist in
one Python environment, and that the network boundary is therefore forced
rather than chosen.

Three rounds say otherwise: 1/3 at the metadata layer, 0/2 at the install
layer on Linux, 0/6 for environment families against each other. A single
`uv pip install` produced 144 packages containing torch 2.7.1, d4rl 1.1,
mujoco-py 2.1.2.14 and cython 0.29.37 together.

The one genuine conflict - lerobot pinning `gymnasium==0.29.1` - is an
ordinary upstream pin, fixable with a PR, which is what makes it *not*
structural.

It is kept here in full, and the project's positioning was revised rather
than the experiment quietly dropped. A fourth round measured what does
survive - deployment footprint - and found it holds for pure simulators
(224M and no GPU, against 6.5G and a GPU) and **does not hold for robomimic**
(7.2G and a GPU either way, because robomimic ships its own deep learning
stack). That counterexample is stated in E1 rather than left for a reader to
find.

## What was measured badly, and corrected

These live in the findings files, not in git history:

- **E5 claimed a 3.1x round-trip improvement** from the scheduler change.
  That came from a single pair of samples. Three round-trip observations of
  the faster setting ranged over 0.53-1.06 ms - too noisy for a ratio. The
  defensible number is **1.76x throughput**, the median of five runs at the
  old 1 ms default against the median of four at the adopted 0.1 ms, with
  non-overlapping ranges. One 0.1 ms repetition was discarded because its
  client failed to connect, which is why that median is over four and not
  five. The larger claim was retracted.
- **E5 warned that the change would raise idle CPU.** It was then measured,
  and it does not. Writing an unverified worry as a warning was wrong.
- **The measurement harness counted failed runs as data**, once reporting
  118879 exchanges/s from a client that had died on connect. It now requires
  the client log to confirm completion before a number is kept.
- **E1's first two robomimic failures were harness defects**, not evidence,
  and are labelled as such rather than counted.
- **E7's conformance server was stricter than the specification it checks**,
  counting a clean client disconnect as a protocol violation on all 45 runs.
  The specification said nothing about a client stopping at all. All three
  were fixed: the client now sends a close frame, SPEC.md gained a section
  saying it should, and the harness downgraded the case to a note.
- **E7's own prediction P2 was not supported**, and is recorded as such
  rather than quietly reworded into one that was. The reformulation that
  does hold is labelled post-hoc.
- **E8 refuted a diagnosis that had already been written up and shipped as
  four pull requests.** A dropped connection was blamed on a CPU-bound learn
  step outlasting the 20 s WebSocket ping. Learn steps of 190 s were then
  measured to close nothing - `learn` runs off the event loop - and the real
  cause was the machine suspending for 1 h 53 min. The fix that rested on the
  wrong cause was withdrawn; the fix that was read out of the code and
  reproduced in a test was kept. Both versions are in the branch history.
- **E14 named a cause, published it, and then refuted it.** The first
  correction reported that the collapse had a located cause and that it was
  ours. The second refuted that cause along with four more, and the document
  is now titled for what it actually establishes: a policy collapses in one
  iteration and five explanations have been ruled out. Both corrections are
  in the file, dated, rather than folded into a revised story.
- **A dtype claim was published on one seed and withdrawn on five.** The
  collapse was said to be specific to bfloat16 and localised to the
  master-weights path, on the strength of a single 9.00x fall. Five seeds
  refuted it: float32 ends 4.59x below its best on one seed and 2.72x on
  another, and on three of five seeds bfloat16 held *better*.
- **`value_loss_coeff` was used as an intervention when it cannot do
  anything.** `freeze_vlm` sets `requires_grad = False` across pi0.5's trunk,
  so the actor's and critic's parameters are disjoint, and Adam's
  per-parameter normalisation cancels a scalar applied to a loss that reaches
  only one of the two groups. It was varied across three settings and read as
  evidence before anyone checked whether it was connected.

Four claims on this page were themselves wrong, and are corrected above
rather than rewritten away. The E5 round-trip spread of 0.53-1.06 ms comes
from three observations, not five. The 1.76x throughput ratio is a median of
five runs against a median of four, not five against five. Three rounds of
E1 disproved the coexistence claim, not four; the fourth measures the
difference that survives, as the E1 section says. And the sentence about
pinned dependency SHAs described all of the scripts when it is true of E1's
alone.

## Pre-registered before the data, then run

E9 and E10 were both written and committed while this project had no hardware
to run them on, which is the only moment a pre-registration costs anything.
Both have since run, and one of the four predictions across them was
falsified and is reported as falsified.

E11 was pre-registered before any success rate existed, and its four
predictions came back split: one confirmed, one that the protocol's own rule
for overlapping intervals says cannot be separated, one falsified - the
fine-tuned policy scored below its baseline rather than above it - and one that
two crashed training runs left untestable.

E14 was pre-registered the day before it ran, with six predictions and five
dated amendments, each amendment written before the data it bears on. One
prediction went vacuous once the policy scored zero, and the amendment saying
so was committed before the evaluation that made it vacuous had finished.

E15 is the counter-example and labels itself as one: its runs were already in
flight when it was written, and two of the five results were known. What it
registers is the rule for reading the other three, which is weaker than a
pre-registration and is described in the file as weaker.

E3 remains pre-registered with no data, and has been for longer than any of
those, which is the honest reason to treat a pre-registration as a commitment
rather than an achievement.

## What is missing

- **A true cross-machine number.** E7 got as far as one virtual machine to
  its host, which separates the network stack from the machine boundary but
  still shares a CPU and a hypervisor. A physical NIC and a switch will cost
  more; how much more is unmeasured, and needs a second computer.
- **E3 was never run.** Its protocol is pre-registered, including an explicit
  declaration of the familiarity bias that would have favoured PlugRL and
  three ranked mitigations, but no data exists.
- **No VLA has been trained through this system without being destroyed by
  it.** The memory problem that stopped E11 at one iteration of ten is fixed
  and is no longer what is missing: E14 ran nine iterations on one card pair
  with no `OutOfMemoryError` at all. What E14 found instead is that the
  policy collapses at the *first* iteration - 29 of 50 to 0 of 50 - and stays
  there for the remaining eight. Ten hypotheses have been refuted, including
  every one that limits how far the policy moves, and the cause is not
  located. **This is a defect in our use of FPO until shown otherwise, not a
  result about FPO**, and the findings files say so. E6 remains the only
  complete learning curve, and it trains a 272k-parameter MLP on continuous
  control.
- **A second algorithm had never run.** Every learning result here is FPO, so
  a defect in the FPO path and a property of this setting cannot be told
  apart. `dppo` was on the CLI menu the whole time and died with a `KeyError`
  when selected.

## Reproducing

Most scripts assume Linux. Several were driven from Windows through WSL, and
the `wsl-*.sh` wrappers set up a proxy and a clean git config before handing
off - adapt or ignore those. E6 runs on Windows or Linux and needs no GPU.

`results*/` holds the logs behind each finding.
