# E29: the log-probability clamp is not why `fpo-policy` stands still under DPPO; its narrow noise is part of it

2026-09-25 · Linux workstation (`guangzhao`), CPU only · `fpo-policy` · DPPO ·
Hopper-v5, three arms, three seeds, 100 iterations · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`20484bf`, after the pilot in
[`results/pilot.txt`](results/pilot.txt))

---

## The result

`fpo-policy` learns Hopper under FPO and not under DPPO (E28), where DPPO's
own policy improves it by +260 to +417. Reading the code found two places
where the two policies meet DPPO's loss differently: the clamp of every
log-probability to [-5, 2], which binds on half of `fpo-policy`'s elements
and none of `dppo-policy`'s, and the per-step noise, about a tenth of
`dppo-policy`'s. E29 changed each, one arm apiece, on E28's cell.

| arm | change | gain, seeds 0 / 1 / 2 | verdict |
| --- | --- | --- | --- |
| `control` | none | +10.6 / +4.5 / +9.7 | P2 holds: E28 was +10.6 / +4.9 / +9.7 |
| `noclamp` | upper bound lifted | **+7.0 / +3.0 / +7.8** | **P3 falsified**: 0 of 3 past +100 |
| `wide` | noise level ×10 | **+73.6 / +84.6 / +106.2** | **P4 falsified**: 1 of 3 past +100 |

(Gain: mean return of iterations 91-100 minus the first iteration's.)

* **P1 holds**, 3 of 3 arms, 9 of 9 seeds; 20 minutes each.
* **V1 passes**: the logged entropy is −1.9198 in `control` and `noclamp`
  and +0.3828 in `wide`, at every iteration - the expected −1.920 and
  −1.920 + ln 10 = +0.383.

**The clamp is ruled out.** Lifting it changes nothing measurable: the gains
sit inside the control's, and the actor moves as much as the control's
(15-20% from iteration 20 to 100, against 20-23%).

**The noise is part of it, and not all of it.** By the registered bar - +100
on two seeds of three - P4 is falsified, and the protocol's reading for two
falsifications, "neither is sufficient", is what is claimed. Described
beside it: every `wide` seed gained seven to nineteen times its control
seed, and every one ended at least 60 above its control seed's best
iteration.
Mean return per ten iterations:

| arm, seed | 1-10 | 21-30 | 41-50 | 61-70 | 81-90 | 91-100 |
| --- | --- | --- | --- | --- | --- | --- |
| `wide` 0 | 17 | 19 | 25 | 56 | 77 | 90 |
| `wide` 1 | 21 | 23 | 30 | 54 | 89 | 106 |
| `wide` 2 | 19 | 22 | 28 | 52 | 97 | 126 |
| `control` 0-2 | 16-18 | 17-19 | 19 | 19-22 | 22-24 | 24-27 |

`wide` barely moves for forty iterations, then climbs, and is still climbing
at the end; episodes grew from about 23 steps to 58-76. At an entropy about
`dppo-policy`'s, it gains about a quarter of what `dppo-policy` gained in
E28.

---

## What I expected, and what the pilot misled me about

After the pilot I registered both as expected to fail. By the rule both did;
about `wide` I was wrong in substance. In eleven iterations `wide` had made
`approx_kl` ten times *smaller* than the control's, and I read that as the
ratio moving less. It is not comparable across noise levels: the KL between
two normals for the same shift of the mean scales as 1/σ², so ten times the
deviation divides it by a hundred for the same change. The actor moved
**40-43%** in `wide` against the control's 20-23%.

---

## What this does and does not support

**Supported:**

* The log-probability clamp does not stop `fpo-policy` learning under DPPO.
  #64 keeps it configurable and its default the reference's; E29 gives no
  reason to change that default.
* Ten times the flow policy's noise, for sampling and for the ratio alike,
  lets DPPO move `fpo-policy`'s return on Hopper by +74 to +106 in 409,600
  steps, where the unchanged policy moves +5 to +11.

**Not supported:**

* That `fpo-policy` learns Hopper under DPPO at any noise level: the bar was
  not met, and one level was tried.
* That the noise is the whole difference from `dppo-policy`. At matched
  entropy the gap is still about fourfold. The two policies also differ in
  network (`fpo-policy`'s four hidden layers of 32 against DPPO's own),
  action chunk (1 against 4), denoising steps (10 against 20) and
  `fpo-policy`'s output scale of 0.25; none is tested here.
* Anything about how the noise helps - through exploration, through the
  variance of the policy gradient, or through the ratio. E29 varies the
  level once and does not separate them.

---

## Reproducing

```bash
bash run.sh                                  # three arms at once; about 20 minutes on 24 cores
python summarise.py                          # P1, V1, P2-P4, the diagnostics; writes summary.tsv
PYTHONPATH=src python clamp_probe.py 0.1     # the clamp's reach on fresh policies, per step
```

`results/verdicts.txt` is `summarise.py`'s output. The pilot's own logs stay
on `guangzhao`; what they showed is in `results/pilot.txt`. Checkpoints and
tensorboard files stay there too (`.gitignore`).
