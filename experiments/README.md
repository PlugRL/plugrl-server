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

The first of those five answers is negative, and it is the one the project
was built on. The last one cost a bug and a retracted diagnosis, both of
which are written down.

| | Question | Answer |
|---|---|---|
| [`e1-dependency-conflict`](e1-dependency-conflict/) | Can a training stack and an environment stack share one Python environment? | **Yes** - the claim that they cannot is disproved |
| [`e2-cross-language`](e2-cross-language/) | Can the protocol be spoken by something that is not this codebase? | **Yes** - an 843-line C++ client with no third-party libraries drives a real server |
| [`e3-integration-cost`](e3-integration-cost/) | What does adding a new environment or policy cost, here versus elsewhere? | **Never run.** The metrics are pre-registered; the plan is kept, the result does not exist |
| [`e5-boundary-cost`](e5-boundary-cost/) | What does crossing the process boundary cost per step? | **Sub-millisecond** on loopback - a lower bound, not a cross-machine number |
| [`e6-first-learning-curve`](e6-first-learning-curve/) | Does anything here actually learn? | **Yes** - three seeds, episode return from about -300 into the thousands |
| [`e7-cross-machine`](e7-cross-machine/) | What does the boundary cost once packets leave loopback? | **+0.52 ms** on a 184 KiB observation - and the cost is in leaving the machine, not in the network stack |
| [`e8-keepalive-hypothesis`](e8-keepalive-hypothesis/) | Does a long learn step kill the WebSocket connection? | **No** - learns of 190 s, nine times the ping timeout, close nothing. The hypothesis was mine and the measurement refuted it |

Each directory has a `FINDINGS.md` stating what was asked, what came back,
and what it does and does not support. Scripts pin their dependency SHAs at
the top, as resolved on the day they ran, so a rerun measures the same thing
rather than whatever the forks have become.

## E1 disproved the hypothesis it was written to test

The starting claim was that a modern training stack (torch 2.7, CUDA) and an
environment stack (`cython<3`, `mujoco-py`, `robosuite<1.5`) cannot coexist in
one Python environment, and that the network boundary is therefore forced
rather than chosen.

Four rounds say otherwise: 1/3 at the metadata layer, 0/2 at the install
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
  That came from a single pair of samples. Five repetitions showed the faster
  setting ranging over 0.53-1.06 ms - too noisy for a ratio. The defensible
  number is **1.76x throughput**, medians of five runs with non-overlapping
  ranges. The larger claim was retracted.
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

## Pre-registered and not yet run

Two protocols are committed with no data behind them, because the hardware
they need has not been available. Writing them now is deliberate: a
pre-registration is worth something only if it predates the data, and the
cleanest moment to fix the rules is when collecting any is impossible.

| | Question | Status |
|---|---|---|
| [`e9-many-clients`](e9-many-clients/) | What does a second, fourth and eighth env client cost one server - and does it stay correct? | Pre-registered. Needs two machines that can reach each other |
| [`e10-vla-forward-cost`](e10-vla-forward-cost/) | What does a VLA forward actually cost, and is the boundary therefore cheap? | Pre-registered. Needs a GPU and the `openpi` extra |

E9 also collects, as its one-client cell on a second machine, **the L2 rung
E7 could not reach**. E10 measures **the denominator E5 borrowed** - the
number that decides whether "the boundary is cheap" is a finding or a hope.

E3 is pre-registered in the same way and has been for longer, which is the
honest reason to say that a pre-registration is a commitment and not an
achievement.

## What is missing

- **A true cross-machine number.** E7 got as far as one virtual machine to
  its host, which separates the network stack from the machine boundary but
  still shares a CPU and a hypervisor. A physical NIC and a switch will cost
  more; how much more is unmeasured, and needs a second computer.
- **E3 was never run.** Its protocol is pre-registered, including an explicit
  declaration of the familiarity bias that would have favoured PlugRL and
  three ranked mitigations, but no data exists.
- **No VLA has ever run through this system.** E6 trains a 272k-parameter MLP
  on continuous control. The openpi policy path exists and has never been
  executed.

## Reproducing

Most scripts assume Linux. Several were driven from Windows through WSL, and
the `wsl-*.sh` wrappers set up a proxy and a clean git config before handing
off - adapt or ignore those. E6 runs on Windows or Linux and needs no GPU.

`results*/` holds the logs behind each finding.
