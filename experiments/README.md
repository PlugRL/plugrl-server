# Experiments

Reproduction scripts and findings. Each directory has a `FINDINGS.md` stating
what was asked, what came back, and what it does and does not support.

| | Question | Answer |
|---|---|---|
| `e1-dependency-conflict` | Can a training stack and an environment stack share one Python environment? | **Yes** - the claim that they cannot is disproved |
| `e2-cross-language` | Can the protocol be spoken by something that is not this codebase? | **Yes** - a 693-line C++ client with no third-party libraries drives a real server |
| `e3-integration-cost` | What does adding a new environment or policy cost, here versus elsewhere? | Not measured yet; the metrics are pre-registered |
| `e5-boundary-cost` | What does crossing the process boundary cost per step? | Sub-millisecond, and the shipped scheduler interval was costing more than the boundary did |

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

This is kept here in full because a disproved hypothesis is a result, and
because anyone building an argument on that claim should see it fail first.

## Reproducing

The scripts assume Linux. Several were driven from Windows through WSL, and
the `wsl-*.sh` wrappers set up a proxy and a clean git config before handing
off - adapt or ignore those.

Dependency SHAs are pinned at the top of each script as of 2026-09-09, so a
rerun measures the same thing rather than whatever the forks have become.

`results*/` holds the logs behind each finding.
