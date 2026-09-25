# E21 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. V1 failed on the first build; the arithmetic is now float64

**2026-09-25, after the first build and before any evaluation.** The protocol
provides for exactly this: *if any check fails, nothing is evaluated; the
construction is fixed and the fix recorded here.*

The first build ([`results/construction-first-build.txt`](results/construction-first-build.txt))
passed V2 and V3 on all nine arms - every arm's action-expert distance within
0.9990 to 0.9993 of its target - and **failed V1**: the `fpo` arm, `θ + Δ`,
differed from E14's first checkpoint in 98 of the 208 tensors.

[`diagnose_v1.py`](diagnose_v1.py), output in
[`results/diagnose-v1.txt`](results/diagnose-v1.txt), found the cause. The
first build computed `Δ = b − a` and `θ + Δ` in float32. A float32 difference
is exact when `b/a` lies in [1/2, 2] and not in general outside it. **Every
one** of the 1,463,176 missed elements - 1,463,158 in 81 float32 tensors, 18
in 17 bfloat16 tensors - is an element FPO's update took below half or above
twice its value, or across zero. There, `a + (b − a)` rounds one step away
from `b`. The same arithmetic in float64 reproduces all 208 tensors bit for
bit.

`make_arms.py` now computes `Δ` and every arm in float64 and casts once to the
stored dtype. The design, the arms, the seeds and every threshold are
unchanged. The size of the defect is visible in the first build's distances,
which agree with the targets to three significant figures: it moved elements
by a step at most, where FPO had already moved them by more than their own
size. The check caught a construction that did not reproduce E14's actor
exactly, which is what it was written to catch.
