# E41: run on to FPO++'s 8M steps, FPO with FPO++'s fine-tuning learns square by the bar - two seeds past it, the third falling back

2026-09-28 · Linux workstation (`guangzhao`), CPU only · E39's three seeds
resumed at iteration 101 and run to 167, 48,000 steps an iteration ·
protocol: [`PROTOCOL.md`](PROTOCOL.md) (`3898cce`, after the pilot in
`results/pilot.txt`, before the registered run)

---

## The result

E39's cell, each seed resumed from its own final checkpoint with
`--algo.restore all` and every setting unchanged. Training-rollout success:

| seed | start: E39's iterations 1-2 | E39's end: 91-100 | end: 158-167 | gain |
| --- | --- | --- | --- | --- |
| 0 | 0.495 | 0.787 | 0.786 | **+0.291** |
| 1 | 0.524 | 0.684 | 0.738 | **+0.214** |
| 2 | 0.518 | 0.711 | 0.672 | +0.153 |

* **P1 holds**: 67 iterations on every seed, seven checkpoints each, no
  traceback, clients exited 0. It took 4.8 hours.
* **V1 passes**: every seed restored its own E39 run with `restore=all`,
  logged 101 as its first iteration, and ran every setting `summarise.py`
  lists.
* **The status rule**: 2 of 3 seeds gained at least 0.2, so the cell
  **learns**, and `fpo-policy` · FPO · square becomes "learns".
* **P2 holds, and its grounds were wrong about which seeds.** They had
  seed 2 crossing within a few iterations and seed 1 needing most of the
  extension. The reverse happened: seed 1 rose 0.05 over the extension and
  crossed, and seed 2 fell 0.04 below where E39 left it.

Fifty-episode evaluations of the final checkpoints, with `eval50.sh`: E39's
script with E41's runs. The clone was not run again; its 0.50 is E39's, from
the same script.

| | E39, 4.8M steps | E41, 8M steps |
| --- | --- | --- |
| the clone | 0.50 | |
| seed 0, final | 0.80 | **0.82** |
| seed 1, final | 0.64 | 0.64 |
| seed 2, final | 0.64 | 0.54 |

---

## The shape of the extension

Success by window of ten iterations, E39's last and then E41's (the last
window is seven iterations):

| seed | 91-100 | 101-110 | 111-120 | 121-130 | 131-140 | 141-150 | 151-160 | 161-167 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.787 | 0.809 | 0.814 | 0.817 | **0.827** | 0.809 | 0.800 | 0.778 |
| 1 | 0.684 | 0.679 | 0.709 | 0.729 | **0.760** | 0.755 | 0.728 | 0.735 |
| 2 | 0.711 | **0.744** | **0.744** | 0.738 | 0.716 | 0.707 | 0.699 | 0.658 |

E39's seeds were all still rising when it stopped. Here each rose for a
while and then came down: seed 0 by 0.05 from its best window, seed 1 by
0.03, seed 2 by 0.09. Seed 2's evaluation fell with it, from 0.64 to 0.54.
The full table from iteration 1 is in `results/verdicts.txt`.

The critic kept fitting the returns: explained variance over the last ten
iterations was 0.71 / 0.75 / 0.68, against E39's 0.66-0.68. The clipped
fraction stayed between 0.21 and 0.34, a little above E39's last ten
iterations.

---

## What E41 does not show

* **That the gain holds.** By the registered rule the cell learns, and that
  is its status. But every seed came down from its best over the last thirty
  to sixty iterations, and whether that is noise or the start of a decline
  is not something this run can separate. As PROTOCOL.md says, there is no
  further extension of this cell under this setting.
* **That a resumed run is an uninterrupted one.** The random streams
  restarted at iteration 101 and the first buffer was collected fresh
  (PROTOCOL.md, known item 3).
* **Which of FPO++'s settings did the work**, as in E39.
* **Anything about pi0.5.** E42 runs FPO++'s fine-tuning in full there.

Checkpoints and tensorboards stay on `guangzhao`. The logs, `verdicts.txt`,
`summary.tsv` and the evaluations are here.
