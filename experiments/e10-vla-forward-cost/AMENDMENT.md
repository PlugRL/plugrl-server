# E10 protocol amendment 1

**Written 2026-09-11, after data collection had started.**

`PROTOCOL.md` says it must not be edited once the first data point exists,
and that a flawed rule is replaced by a new document saying why the old one
was unusable, with both kept. Collection has started. **`PROTOCOL.md` is
therefore left exactly as written**, including the sentence corrected below,
and this file carries the corrections.

Two of the three entries are corrections to the protocol. The third is a
change to the design, recorded before the results rather than explained
after them. Sibling findings files are cited by section and quotation rather
than by line number, because they are being edited while this is written.

---

## Correction 1: the motivating arithmetic uses a figure the same paragraph supersedes

`PROTOCOL.md` lines 15-22 say, in one paragraph, both of these:

> The measurement behind that word is 0.5 ms on loopback, and E7 raised it to
> 1.3 ms across a machine boundary.

> at a 30 ms forward the boundary is under 2% of a step, at a 2 ms forward it
> is a fifth of one

**The two percentages only hold at 0.5 ms.** Worked out:

| boundary | forward | boundary / (boundary + forward) | boundary / forward |
|---|---|---|---|
| 0.5 ms | 30 ms | 1.6% | 1.7% |
| 0.5 ms | 2 ms | 20.0% ("a fifth") | 25.0% |
| 1.3 ms | 30 ms | 4.2% | 4.3% |
| 1.3 ms | 2 ms | 39.4% | 65.0% |

E7's figure is a median of 1.304 ms with a range of [1.220, 1.359] over its
repetitions at the 184 KiB payload, summarised in its "What this does and
does not support" section as "The process boundary costs about 1.3 ms".
Metric 2 of this protocol computes the ratio from that 1.3 ms figure, and P2
uses it.

**The correct sentence is:** at a 30 ms forward the boundary is about 4% of a
step, at a 2 ms forward it is two fifths of one.

The sentence was inherited verbatim from the blockquote in
`e5-boundary-cost/FINDINGS.md` under "The boundary is cheap", where it is
about E5's own loopback number and is correct in that context. Carrying it
into E10 unchanged understates the stakes of the measurement E10 exists to
make: at 1.3 ms the unfavourable case is worse, not better. Nothing in the
metrics, the predictions or the stopping rule depends on the wrong pair of
percentages, so no result changes. The motivation was underclaimed.

## Correction 2: the policy table does not describe the policy that ships

`PROTOCOL.md` lines 35-39 list `pi0` at "~3B" as the second and third levels
of the independent variable. The default the server actually ships is
`pi05_tiny_libero`
(`plugrl-server/src/plugrl_server/policy/openpi/openpi_policy.py:38`), which
`third_party/openpi/src/openpi/training/config.py:756-757` builds with
`paligemma_variant="gemma_tiny"` and
`action_expert_variant="gemma_expert_tiny"`. Those two variants are width
2048 / depth 6 and width 1024 / depth 6
(`third_party/openpi/src/openpi/models/gemma.py:109-126`), against the
`gemma_2b` / `gemma_300m` pair at `gemma.py:69-87` that the "~3B" figure
describes. The table's row and the shipped configuration are different
models.

**The run reports 1.71 billion parameters for `pi05_tiny_libero`**, not ~3B.

**That count is not reproducible from what is committed in this working tree,
and is recorded as reported rather than as verified.** `results/` does not
exist yet, and the `openpi` extra is not installed in any of the four
checkouts - `jax`, `flax` and `transformers` are all absent from
`plugrl-server/.venv` - so the model cannot be constructed here to recount
it. When the results land, 1.71e9 belongs in the `params` column of
`results/summary.tsv` and should be checked against it. If the two disagree,
the file wins and this paragraph is wrong.

What is verifiable from committed files, and is the part that matters, is the
structural claim: the shipped default is a tiny variant, and the protocol's
table attached a parameter count from a different model to it.

## Change 1: a second policy level, added and labelled

A second level of the independent variable was added during the run: the
**full-size variant**, that is `Pi0Config`'s own defaults
`paligemma_variant="gemma_2b"` and `action_expert_variant="gemma_300m"`
(`third_party/openpi/src/openpi/models/pi0_config.py:20-21`), which is the
model the "~3B" in the protocol's table refers to.

The reason is correction 2. Measuring only the shipped tiny default answers a
question about a model the project's cost argument does not cite; measuring
only the full-size variant says nothing about what a user running the shipped
configuration actually pays. Both are needed, and the ratio in metric 2
differs between them.

This follows E7's precedent, which added a third rung to a two-rung protocol
and recorded it under "Two deviations from the protocol, both deliberate" as
"A third rung was added", rather than folding it in silently. As there, the
addition is stated before the numbers, and every row must carry which level
it came from - the `policy` column of the record format already does that,
and no row is to be reported without it.

## What this does not change

The question, the held-fixed list, metrics 1 to 3, P1 to P3 as worded, the
repetition and validity rules, the stopping rule and the record format all
stand as pre-registered. Correction 1 changes a motivating percentage,
correction 2 changes a parameter count in a table, and change 1 adds a level
to one variable.
