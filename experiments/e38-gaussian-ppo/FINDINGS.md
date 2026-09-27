# E38: a Gaussian policy with PPO, run as CleanRL runs it, learns all three MuJoCo tasks through PlugRL - at CleanRL's returns

2026-09-27 · Linux workstation (`guangzhao`), CPU only · three cells, three
seeds, 488 iterations of 2,048 steps · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(`2f70733`, after the pilot and the GAE diagnostic in `results/pilot.txt`,
before the registered run)

---

## The result

`gaussian-policy` under `ppo` (#85) is CleanRL's `ppo_continuous_action.py`
with every default, on code with the GAE episode-boundary fix (#86). The
coverage figure's rule, read on iterations 479-488:

| cell | seed | first iteration | iterations 479-488 | CleanRL (-v4) |
| --- | --- | --- | --- | --- |
| HalfCheetah | 0 | -335.2 | 1512.9 (**+1848**) | 1442.64 +/- 46.03 |
| | 1 | -384.6 | 1442.5 (**+1827**) | |
| | 2 | -393.9 | 1557.8 (**+1952**) | |
| Hopper | 0 | 12.1 | **2298.7** | 2382.86 +/- 271.74 |
| | 1 | 15.2 | **2207.4** | |
| | 2 | 14.2 | **2181.3** | |
| Walker2d | 0 | -0.7 | **2957.2** | 2287.95 +/- 571.78 |
| | 1 | -1.8 | **3260.0** | |
| | 2 | -0.6 | **3154.2** | |

* **P1 holds**: nine runs, each with 488 iterations logged, 24 checkpoints,
  no traceback and a client that exited 0. The three cells ran at once
  beside E37 and took 28 minutes.
* **V1 passes**: every iteration of every run learned at CleanRL's annealed
  rate, from 3e-4 down to 6.15e-7 (3e-4 / 488) at the last.
* **P2, P3, P4 hold**: it learns Hopper, Walker2d and HalfCheetah, each on
  3 of 3 seeds. The smallest figure is 4.4 times its bar.

The new row, `gaussian-policy` · PPO, **learns** on all three tasks.

---

## Beside CleanRL

Mean return over three windows:

| cell | seed | 91-100 | 241-250 | 479-488 |
| --- | --- | --- | --- | --- |
| HalfCheetah | 0 | 744.4 | 1322.5 | 1512.9 |
| | 1 | 999.0 | 1348.1 | 1442.5 |
| | 2 | 925.5 | 1428.5 | 1557.8 |
| Hopper | 0 | 1623.2 | 2242.8 | 2298.7 |
| | 1 | 1297.1 | 2486.9 | 2207.4 |
| | 2 | 920.3 | 2440.6 | 2181.3 |
| Walker2d | 0 | 514.9 | 2747.0 | 2957.2 |
| | 1 | 476.6 | 2037.8 | 3260.0 |
| | 2 | 652.6 | 3344.2 | 3154.2 |

HalfCheetah ends at CleanRL's number or a little above it, and Hopper
within its spread. Walker2d ends above it, by more than CleanRL's own
spread; E38 does not explain that. The protocol registered these as reported
figures, not a test. The tasks are -v5, not -v4, and three things are done
differently (declared there). What they do show is that the one pair here
whose behaviour is known well outside this project comes out of PlugRL
about where it comes out of CleanRL. Here the server has no simulator
installed, and the environment runs in a separate client with only the
protocol between them.

Episodes grew from 18-20 steps to 586-636 on Hopper and 686-792 on Walker2d.
HalfCheetah's are always 1,000.

---

## What E38 does not show

* **That the platform was right before #86.** E38 ran with the GAE fix. The
  diagnostic in `results/pilot.txt` ran this pair on Hopper for 100
  iterations on each side of the fix. By iteration 100 the two were
  indistinguishable on three seeds (951 against 941 on average), so the
  defect cost this task little. That diagnostic was not registered.
* **The square column.** `gaussian-policy` was not run on robomimic square.
  From random weights, square's sparse reward gives it nothing to learn from
  (E27), and no behaviour-cloned Gaussian start was built.

Checkpoints and tensorboards stay on `guangzhao`. The logs, `verdicts.txt`
and `summary.tsv` are here.
