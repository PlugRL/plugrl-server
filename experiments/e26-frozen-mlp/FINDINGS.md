# E26: freezing the action expert's MLP does not stop FPO destroying pi0.5

2026-09-25/26 · qz103 · `pi0-policy` `pi05_libero`, LIBERO-10 task 8, one FPO
iteration per arm, fifty-episode evaluations · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (before any run) · code `main` + #60 (4d7ee38)

---

## The result

E22 found that FPO's first update, restricted to the action expert's 54 MLP
matrices, takes pi0.5 to 0 of 50, while restricted to attention, the
modulation or the projections it leaves 24 to 26. E26 asked whether that
located a remedy: train with the MLP frozen, and does the first iteration
still destroy the policy?

**It does.**

| arm | trained | successes of 50 | steps per episode |
| --- | --- | --- | --- |
| `base` | no | **29** (0.58, 95% CI 0.44-0.71) | 457 on average |
| `control` | one FPO iteration | **0** (CI 0-0.07) | 520, every one |
| `frozen` | one FPO iteration, expert MLP frozen | **0** (CI 0-0.07) | 520, every one |

* **V1 passes**: in `frozen`'s checkpoint all 54 MLP tensors are
  bit-identical to the base and every other expert tensor moved; in
  `control`'s every MLP tensor moved. The frozen server logged "Froze 54
  action-expert tensors matching `\.mlp\.`".
* **V2 passes**: three evaluations, fifty episodes each, client exit 0,
  episodes and steps reconciled between client and server.
* **P1 holds**: the collapse reproduces on today's code - `control` 0 of 50,
  every episode run to the 520-step limit, as in E14.
* **P2 falsified**: `frozen` scores **0 of 50**, every episode to the limit
  too. By the protocol: the rest of the expert, trained, destroys the policy
  as well.
* **Reported**: `base` scores 29 on this code, inside the older null of
  28-37.

---

## Where the damage went

Relative distance from the base, per module group, after the one iteration:

| group | `control` | `frozen` | E14's first iteration |
| --- | --- | --- | --- |
| modulation (74) | 2.17% | **2.97%** | 2.11% |
| attention (72) | 1.71% | **2.42%** | 1.66% |
| MLP (54) | 1.59% | 0 | 1.55% |
| projections (8) | 1.54% | **1.96%** | 1.32% |

`control` moves the expert as E14 did, group by group. With the MLP held,
FPO moves each remaining group 27-41% further, and those three together are
enough. E22 had only ever added FPO's update to one group at a time; the
three non-MLP groups together were never tested. So E22's localisation holds
for the perturbation it built - FPO's own update, restricted after the fact
- and does not carry over to training: freeze where the damage was, and
training puts it somewhere else.

What stands from E21 and E22 is that it is the **direction** of FPO's update,
not its size, that destroys pi0.5. What E26 removes is the hope that the
direction lives in one module group that can simply be switched off.

---

## What this does and does not support

**Supported:**

* On today's code, one FPO iteration still takes pi0.5 from 29 of 50 to 0 on
  LIBERO-10 task 8.
* Freezing the action expert's MLP (#60, `--policy.freeze-expert-params
  '\.mlp\.'`) does not prevent it.

**Not supported:**

* That any other single group, frozen, would not prevent it - only the MLP
  was tried.
* That the three non-MLP groups destroy the policy *because* they moved
  further: freezing changes the gradient every other group receives, and E26
  does not separate "moved more" from "moved differently".
* Anything past the first iteration, or about another task or seed.

**A caveat on the evidence of code.** The evaluation harness writes a source
manifest (`environment-source-e26-*.txt`) by hashing the server's source
directory. `derive.py` pointed the code at `$R/plugrl-server-e26` through
`PYTHONPATH`, and the manifest was hashed without it, so the three manifests
describe the older copy, not the code that ran. What does show the code that
ran: `run.log`'s `code: 4d7ee38` (the deployment's marker), the `Froze 54`
line, which only #60's code can print, and V1's bit-identical MLP.

---

## Reproducing

On the cluster, with this directory copied to `$R/e26` and #60's code
deployed LF at `$R/plugrl-server-e26`: `setsid nohup bash run.sh > run.out
2>&1 &` - about two hours, the base evaluation and both training runs in
parallel, then the two evaluations. `train.sh` and `eval.sh` are committed as
`derive.py` wrote them from E14's. Then, here:

```bash
python summarise.py   # V1, V2, P1, P2, base and the movements, from results/
```

`results/` holds `check-frozen.txt`, the harness's tables, every training and
evaluation log, and both training servers' logs. The checkpoints stay on the
cluster.
