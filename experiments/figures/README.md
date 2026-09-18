# Figures

Every figure here is generated from a result table committed in this
repository, by one script:

```bash
python experiments/figures/make_figures.py
```

It reads nothing but those tables, writes the four SVGs next to itself, and
prints what it wrote. Regenerate them after a results file changes; a figure
that disagrees with its table is a bug in the script, not a matter of taste.

| figure | drawn from |
|---|---|
| `e6-learning-curve.svg` | [`e6-first-learning-curve/summary.tsv`](../e6-first-learning-curve/summary.tsv) |
| `e11-stageA-control.svg` | [`e11-vla-rl-libero/results/stageA.tsv`](../e11-vla-rl-libero/results/stageA.tsv) |
| `e11-stageC-result.svg` | [`e11-vla-rl-libero/results/stageC_eval.tsv`](../e11-vla-rl-libero/results/stageC_eval.tsv) |
| `e11-time-breakdown.svg` | [`e11-vla-rl-libero/FINDINGS.md`](../e11-vla-rl-libero/FINDINGS.md), "Where an hour of this training goes" |

The timing figure is the one exception to "read from a table": its seconds are
transcribed into the script, with the source named there. It plots seconds
rather than shares, because that file's share column for the learn's three
components is computed against a denominator it does not state - 1,389 s is
44.6% of 3,114 s, not of the learn's 3,490 s. The 472 s that the components do
not account for is drawn as its own segment instead of being dropped.

Requires `matplotlib`; no other third-party dependency.
