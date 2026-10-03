# E57 amendment

## 2026-10-03 00:25 EDT: summarise.py read a blank cell as a number

All 834 registered evaluations had finished and none had failed, before
anything was summarised. Then `summarise.py` stopped with
`ValueError: invalid literal for int() with base 10: ''`.

The cause is the six runs of E54's two loud faults, `indices:reorder` and
`reward:short-array`. They raised an error, logged no return, and so have a
blank `strict` in E54's `runs.csv`. `summarise.py` converted `rule` and
`strict` to integers without allowing for a blank.

The fix is in one line: a blank cell becomes `None`. A run with `None` is
not "silent on the curve". The protocol leaves the loud faults out of every
count, so the fix changes no count, no rule and no prediction. Nothing else
in `summarise.py`, `verdicts.py` or `evaluate.py` changed. No result had been
seen when the fix was made: the crash came before any output.
