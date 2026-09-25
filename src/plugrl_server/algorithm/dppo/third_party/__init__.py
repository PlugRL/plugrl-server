"""Code from other projects, kept here so DPPO does not need theirs installed.

`plugrl-server[dppo]` pins a fork of the DPPO repository as a git dependency.
The DPPO *algorithm* in this package reached into it for exactly two utilities
- a running mean/variance and a learning-rate schedule - and neither is
DPPO-specific: one comes from OpenAI's phasic-policy-gradient, the other from
a standalone scheduler package. Both are MIT licensed.

Carrying those two here means `dppo` and `dppo-dist` run on a plain install.
The extra is still needed for `dppo-policy`, which wraps DPPO's own
`DiffusionModel` and builds it through DPPO's hydra configs; that is a real
dependency on the project rather than on two utility classes.

Each file keeps its upstream path name and its licence header, so a reader can
diff it against the original.
"""
