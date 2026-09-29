# E43 amendments

## 1 - 2026-09-28, after the registered run: three more seeds per arm

**What prompted it.** The registered run holds on every prediction. But the
two arms ended apart:
- The cross arm's last ten iterations averaged 2,060, 2,127 and 946.
- The local arm's averaged 1,262, 695 and 725.

The figure shows that gap from about iteration 40 on. PROTOCOL.md predicted
nothing about it. The two arms should draw from the same distribution:
- the same code
- the same settings
- the same seeds
- trajectories that differ only by floating-point bits (known item 4)

A real gap would then be a defect to find, not a result. With three seeds per
arm, luck cannot be told from a difference.

**What is added.** Seeds 3, 4 and 5 in both arms, with the same scripts, the
same concurrent layout and the same machines. They are written to
`results/local-b`, `results/cross-b` and `results/cross-clients-b`.

**How it is read.** The additions are reported, not predicted. P1-P6 stay as
registered, on seeds 0-2. The six seeds per arm are compared on:
- iteration 91-100's mean
- the status rule
- the extra time, as P4 measured it

**What would count as a gap worth chasing:** all six cross seeds above all
six local seeds, or the arms' six-seed means further apart than the spread
within either arm. Anything less is read as seed variation.
