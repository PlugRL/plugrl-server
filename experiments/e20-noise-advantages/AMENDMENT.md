# E20 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. P1 is going to fail, and how P2 will be reported when it does

**2026-09-25 12:44, written after the `reward` arm's first iterations were
visible and before the `zero` arm had started.**

P1 asks the `reward` arm to reproduce E16's `except-critic` gains to within
1.0, on the premise, stated as known item 2, that one-client training is
deterministic here. That premise was mine and it was an overclaim. Its
evidence was only that two runs' **first** iterations matched to the decimal.

The `reward` arm, part way through, against E16's arm on the same seed:

| seed | iterations identical | first divergence |
| --- | --- | --- |
| 0 | 1-2 | iteration 3 |
| 1 | 1-3 | iteration 4 |
| 2 | 1 | iteration 2: 1025.2 here, 1301.2 in E16 |

Same code - this branch is cut from E16's - same checkpoint, same restore
mode, same seed. **Collection is deterministic and learning is not.** Two runs
agree until the first update and then drift apart, by 21% after one update on
seed 2. Why is not established; parallel CPU reductions summing in an order
that depends on thread scheduling are the usual cause, and contention between
concurrent processes makes that scheduling vary.

So P1 will be falsified, and by the protocol P2 is then **reported and not
read as a result**. Declared now, before any `zero` data exists, so the
description cannot be chosen from it:

* The `zero` arm's `end − start` is reported per seed, as P2 would have read
  it, with the words *not a registered result* beside it.
* Beside it goes the run-to-run noise this experiment has turned out to
  measure: the `reward` arm and E16's `except-critic` arm are the **same
  condition run twice**, so the per-seed difference of their `end − start` is
  the spread of two identical runs.
* Nothing further is claimed from the comparison. If the `zero` arm falls far
  outside that spread, the document says so and says that it is a description.
  A registered test of the mechanism needs the determinism fixed first, or
  enough seeds that run-to-run noise is averaged rather than assumed away.

This is the same overclaim that E18's draft, and a comment on #49, rested on.
Both have been corrected.
