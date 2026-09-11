# Experiments

PlugRL splits reinforcement learning training from the environments it learns
from, into two processes joined by a WebSocket protocol. That is a design
choice, and design choices are usually defended with assertions. These ask
instead:

1. **Is the premise true?** The split was justified by a claim that the two
   dependency stacks cannot coexist. → **E1: no, they can.**
2. **What does it cost?** Every step pays for serialization and a round trip.
   → **E5: 0.5-0.7 ms on loopback**, and the shipped scheduler interval was
   costing more than the boundary itself.
3. **Is it actually portable?** A protocol decouples nothing unless something
   other than this codebase can speak it. → **E2: yes** - two clients written
   from the specification alone, one of them C++ with no third-party
   libraries.
4. **Does any of it train?** → **E6: yes** - FPO on HalfCheetah-v5, three
   seeds.

One of those four answers is negative, and it is the one the project was
built on.

| | Question | Answer |
|---|---|---|
| [`e1-dependency-conflict`](e1-dependency-conflict/) | Can a training stack and an environment stack share one Python environment? | **Yes** - the claim that they cannot is disproved |
| [`e2-cross-language`](e2-cross-language/) | Can the protocol be spoken by something that is not this codebase? | **Yes** - an 843-line C++ client with no third-party libraries drives a real server |
| [`e3-integration-cost`](e3-integration-cost/) | What does adding a new environment or policy cost, here versus elsewhere? | **Never run.** The metrics are pre-registered; the plan is kept, the result does not exist |
| [`e5-boundary-cost`](e5-boundary-cost/) | What does crossing the process boundary cost per step? | **Sub-millisecond** on loopback - a lower bound, not a cross-machine number |
| [`e6-first-learning-curve`](e6-first-learning-curve/) | Does anything here actually learn? | **Yes** - three seeds, episode return from about -300 into the thousands |

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

## What is missing

- **E5 is loopback only**, which is a lower bound. The cross-machine number -
  the one the architecture's cost argument actually needs - has never been
  measured.
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
