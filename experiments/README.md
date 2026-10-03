# Experiments

PlugRL's claim is that training should not care where its environments run:
in another process, on another machine, on a machine with no GPU, or in a
program that is not Python, and whatever trainer is on the other side. These
fifty-two experiments test that claim, measure what it costs, and record the
defects found on the way. Fifty-one have run. E3 has a protocol and no data.

Each directory has a `FINDINGS.md`: what was asked, what came back, and what
the result does **not** support. The logs behind it are under `results*/`.
Forty-five of the experiments that ran have a `PROTOCOL.md` with their
predictions. Most were written before the data existed; the exceptions
(E15 was written with its runs in flight) say so. Anything changed after a
protocol is dated in an `AMENDMENT.md`, not folded into it.

## Where the environment side can run

| | Question | Answer |
|---|---|---|
| [`e43-cross-machine-training`](e43-cross-machine-training/) | Does training still work with the env clients on another physical machine? | **Yes.** The quickstart pair learns with its clients on a Windows laptop over campus Wi-Fi, six seeds against six on one machine |
| [`e44-cpp-pendulum`](e44-cpp-pendulum/) | Does an env client have to be Python? | **No.** A C++ program with no third-party libraries trains a policy on its own Pendulum as well as the Python env client does, in less time |
| [`e2-cross-language`](e2-cross-language/) | Can the protocol be spoken from the specification alone? | **Yes.** Two clients written against SPEC.md, one of them C++, drive a real server. They share an author with the specification, so this shows it is sufficient, not that it is clear |
| [`e12-cuda-free-rollout`](e12-cuda-free-rollout/) | Does a rollout machine need CUDA? | **No.** That was a packaging default. The CPU build of the same torch takes robomimic's env client from 7.2G with 16 nvidia wheels to 2.7G with none, and LIBERO's from 7.8G to 3.4G, sending byte-identical observations |
| [`e13-gpu-free-rendering`](e13-gpu-free-rendering/) | Or a GPU to render on? | **No, at 1.91x the wall clock.** Ten clients rendering on the CPU, 30 of 30 episodes successful. Its prediction that the queue would keep this under 1.3x is falsified |
| [`e45-env-side-footprint`](e45-env-side-footprint/) | What do other systems that train through a channel install on the environment side? | **Torch.** RLlib's external-env client and LeRobot's HIL-SERL actor take 6.1 GB each, 27x plugrl-env-client. dm_env_rpc and openpi-client are thin too, but do not train |
| [`e1-dependency-conflict`](e1-dependency-conflict/) | Was the split forced, because the two dependency stacks cannot coexist? | **No.** They install together in one environment. The project's positioning was revised, not the experiment dropped |

## What the split costs

| | Question | Answer |
|---|---|---|
| [`e5-boundary-cost`](e5-boundary-cost/) | What does crossing the process boundary cost per step? | **Under a millisecond on loopback**, and the scheduler's default interval was wasting about as much again |
| [`e47-scheduler-wakeup`](e47-scheduler-wakeup/) | Was that interval fixed? | **Not until now.** Its 0.1 ms sleep became a wait of up to 1 ms in epoll. Waking on work answers an infer in 0.96 ms instead of 2.21, idles at 0.35% of a core, and trains the same weights, 1.87x faster |
| [`e7-cross-machine`](e7-cross-machine/) | And once packets leave the machine? | **+0.52 ms** on a 184 KiB observation, from a virtual machine to its host. The cost is in leaving the machine, not in the network stack |
| [`e43-cross-machine-training`](e43-cross-machine-training/) | And between two physical machines? | **About 3 ms plus twice the observation's bytes over the link**, per exchange: 21 ms for 184 KiB at 18 MB/s over Wi-Fi |
| [`e46-reuse-feedback-obs`](e46-reuse-feedback-obs/) | Can the second copy of the observation go? | **Yes.** With `reuse-feedback-obs` the size term halves, 18.5 to 11.6 ms at 184 KiB, and the same training run gives bit-identical weights. On the way: the server's scheduler sleep costs up to 1 ms per infer |
| [`e10-vla-forward-cost`](e10-vla-forward-cost/) | Is that cheap beside a VLA forward pass? | **On a fast link.** Through openpi's compiled path a forward takes 34.9 ms for the policy PlugRL ships and 100 ms for full-size pi0.5. A 1.3 ms crossing is 1.3-3.6% of that; E43's over Wi-Fi is 21% of the 100 ms, and E46's, with each observation sent once, 12% |
| [`e9-many-clients`](e9-many-clients/) | Does one server stay correct with eight clients feeding it? | **Yes.** Twelve runs at 1, 2, 4 and 8 clients, every one exact to the unit. Throughput was still rising at eight, which falsified its prediction |
| [`e8-keepalive-hypothesis`](e8-keepalive-hypothesis/) | Does a long learn step kill the connection? | **No.** Learn steps of 190 s close nothing. The drop that prompted it was the machine suspending, and a real reconnect bug was found and fixed anyway |

## Trainers that were not written for it

Through `plugrl-bridges` (not public yet): one lockstep server, with
adapters for RLinf, gymnasium's `VectorEnv` and Stable-Baselines3's
`VecEnv`.

| | Question | Answer |
|---|---|---|
| [`e48-rlinf-bridge`](e48-rlinf-bridge/) | Does a trainer from outside this codebase train PlugRL env clients? | **Yes.** RLinf learns HalfCheetah with its clients on its own machine, and on a laptop with no torch, Ray or RLinf, three seeds each. RLinf never closes its environments, so remote clients never hear the run end |
| [`e49-multi-trainer`](e49-multi-trainer/) | And other trainers, through the interface each already has? | **Yes.** RLinf, Stable-Baselines3 and CleanRL each learn, three seeds per cell, a C++ client with no third-party library included. RLinf learns Pendulum late, past E44's budget |
| [`e50-boundary-transparency`](e50-boundary-transparency/) | Does the boundary change what is learned? | **No.** On one machine, SB3 ends on byte-identical weights whether its environments run in its own process or behind the bridge. The bridge adds 0.37-0.58 ms per vector step |
| [`e51-episode-accounting`](e51-episode-accounting/) | Do the episode counters these trainers rely on count right? | **Not all.** Gymnasium 1.3's vector `RecordEpisodeStatistics` drops each episode's first reward under SAME_STEP autoreset. CleanRL's logging logs nothing under gymnasium 1.x's default. SB3, the bridge and the env client count right |
| [`e52-latency-sweep`](e52-latency-sweep/) | Does latency change what is learned? | **No, only the time.** With 0 to 25 ms added each way between client and bridge, every run ends on the in-process weights. Each step costs two one-way delays plus up to 3.1 ms, a little over the predicted band at 25 ms |
| [`e53-image-observations`](e53-image-observations/) | And with camera frames and a CNN? | **Still the same weights.** SB3's CnnPolicy on Atari frames ends byte-identical in process and through the bridge; 16 frames a step cost 2.6-3.0 ms |

## What trains through it

These are the sixteen cells [on the project page](https://plugrl.github.io/#what-runs-on-it),
and the experiments behind them.

| | Question | Answer |
|---|---|---|
| [`e6-first-learning-curve`](e6-first-learning-curve/) | Does anything learn? | **Yes.** FPO on HalfCheetah, three seeds, from about -300 into the thousands |
| [`e23-hopper`](e23-hopper/) | Where episodes end because the agent falls? | **Yes.** FPO learns Hopper |
| [`e24-coverage`](e24-coverage/) | Does every MuJoCo combination the code allows run? | **Yes**, and FPO learns Walker2d |
| [`e38-gaussian-ppo`](e38-gaussian-ppo/) | Does a standard baseline reach its published returns through the split? | **Yes.** A Gaussian policy with PPO, run as CleanRL runs it, learns all three MuJoCo tasks at CleanRL's returns |
| [`e17-second-algorithm`](e17-second-algorithm/) | Does a second algorithm run? | **It runs, and does not learn** in the setting first given to it. Normalising its inputs helps a little ([E18](e18-dppo-normalised/)) |
| [`e28-dppo-learns`](e28-dppo-learns/) | Does it learn in DPPO's own setting? | **Yes.** `dppo-policy` learns HalfCheetah and climbs Hopper and Walker2d; run three times as long, it learns both ([E31](e31-dppo-longer/)) |
| [`e30-fpo-dppo-factors`](e30-fpo-dppo-factors/) | Why did `fpo-policy` stand still under DPPO? | **Its network**, its narrow noise among it; not the log-probability clamp ([E29](e29-clamp-or-noise/)). With DPPO's structure it learns all three tasks ([E33](e33-fpo-dppo-structure/)) |
| [`e27-robomimic`](e27-robomimic/) | Beyond MuJoCo? | **Yes.** The MLP policies run on robomimic |
| [`e34-square-dppo`](e34-square-dppo/) | Fine-tuning in DPPO's own setting? | **Yes.** `dppo-policy` learns robomimic square on PlugRL |
| [`e40-square-gaussian-ppo`](e40-square-gaussian-ppo/) | And the Gaussian baseline there? | **Yes.** DPPO's Gaussian MLP learns square under DPPO's own Gaussian PPO |
| [`e35-square-bc`](e35-square-bc/) | Can `fpo-policy` get a pretrained start on square? | **Yes.** Cloned on DPPO's demonstrations, it succeeds 0.44 of the time |
| [`e37-square-fpo-policy`](e37-square-fpo-policy/) | What does RL do from that start? | **It learns square under DPPO and loses ground under our FPO** |
| [`e41-square-fpo-plus-plus-8m`](e41-square-fpo-plus-plus-8m/) | And with the rest of FPO++'s fine-tuning? | **FPO learns square by the bar** at FPO++'s 8M steps: two seeds past it, the third falling back. At 4.8M it lifts the policy short of the bar ([E39](e39-square-fpo-plus-plus/)) |

## pi0.5, and a collapse that was ours

The first FPO iteration through PlugRL took pi0.5 from 29 of 50 to 0 of 50,
and it stayed there. The project's rule was to treat that as our defect until
shown otherwise. It was two:
- **The loss.** FPO scored an action chunk over all 320 of its elements,
  mostly padding and steps the client never ran (E32).
- **The rest of the fine-tuning.** FPO ran without the fine-tuning FPO++ is
  published with. E37-E39 showed that on square before E42 showed it on
  pi0.5.

With both fixed, FPO holds pi0.5 but has not made it better.

| | Question | Answer |
|---|---|---|
| [`e11-vla-rl-libero`](e11-vla-rl-libero/) | Can a full-size VLA be trained through the boundary? | **Trained, not helped.** pi0.5 ran end to end, the server's counts exact against the clients'. One FPO iteration took the task from 26 of 50 to 0, and a second did not fit in memory |
| [`e14-ten-iterations`](e14-ten-iterations/) | Over ten iterations? | **It collapses in the first**, 29 to 0 of 50, and stays there. Five explanations were refuted |
| [`e15-trust-region`](e15-trust-region/) | Is it about how far one update moves? | **No.** Under 1% of relative movement in the action expert is enough, and halving it does not halve the damage |
| [`e16-critic-restart`](e16-critic-restart/) | Is it the untrained value head? | **No.** On HalfCheetah a random value head slows FPO down, and every arm still rose |
| [`e19-dppo-finetune`](e19-dppo-finetune/) | Does DPPO collapse a good HalfCheetah policy restarted with a fresh value head? | **No, and it barely moves it.** Nor does FPO driven by a random critic's noise ([E20](e20-noise-advantages/)) |
| [`e21-perturbation-direction`](e21-perturbation-direction/) | What about the update does the damage? | **Its directions, not its size** |
| [`e22-subspace-or-concentration`](e22-subspace-or-concentration/) | Where? | **The action expert's MLP.** FPO's update applied there alone scores 0; applied to any other part, 24-26. Freezing that MLP does not help: trained with it frozen, pi0.5 still falls to 0 ([E26](e26-frozen-mlp/)) |
| [`e25-pi0-dppo`](e25-pi0-dppo/) | Does DPPO train pi0.5? | **End to end, and two iterations leave it intact**, because they barely move it |
| [`e32-pi0-fpo-plus-plus`](e32-pi0-fpo-plus-plus/) | Is it the loss? | **Yes, the first defect.** Scored as FPO++ scores an action chunk, one update leaves pi0.5 at 33 of 50 |
| [`e36-pi0-longer`](e36-pi0-longer/) | Over more iterations? | **It still falls**, to 5 and 0 of 50 in five FPO iterations, with the rest of FPO still on its own defaults. Ten DPPO iterations leave it at 20 |
| [`e42-pi0-fpo-plus-plus`](e42-pi0-fpo-plus-plus/) | With FPO++'s fine-tuning in full? | **It holds**: 40 and 27 of 50 after ten iterations on two seeds. Neither reaches the bar of 42, and the lower seed drifts the way E36 fell, far more slowly |

## What was measured badly, and corrected

These live in the findings files, not in git history.

- **E5 claimed a 3.1x round-trip improvement** from the scheduler change,
  from a single pair of samples. The defensible number is **1.76x
  throughput**: the median of five runs at the old 1 ms default against the
  median of four at 0.1 ms, with ranges that do not overlap. Its warning that
  idle CPU would rise was then measured, and it does not.
- **The measurement harness counted failed runs as data**, once reporting
  118,879 exchanges/s from a client that had died on connect. It now requires
  the client log to confirm completion before a number is kept.
- **E7's conformance server was stricter than the specification it checks**,
  counting a clean client disconnect as a violation on all 45 runs. The
  client now sends a close frame, SPEC.md says it should, and the harness
  reports the case as a note.
- **E8 refuted a diagnosis that had already shipped as four pull requests.**
  A dropped connection was blamed on a learn step outlasting the WebSocket
  ping. The real cause was the machine suspending for 1 h 53 min. The fix
  that rested on the wrong cause was withdrawn.
- **E10's 1.3-3.6% assumed E7's link.** Over E43's Wi-Fi the same
  observation costs 21% of pi0.5's forward. The row above gives both.
- **E14 named a cause, published it, and refuted it**, along with four more.
  Both corrections are dated in the file.
- **A dtype claim was published on one seed and withdrawn on five.**
- **`value_loss_coeff` was varied as an intervention when it cannot do
  anything.** With pi0.5's trunk frozen, the actor's and critic's parameters
  are disjoint, and Adam's per-parameter normalisation cancels a scalar on a
  loss that reaches only one of them.
- **The pi0.5 collapse was reported as unexplained through E36.** It was
  two defects of ours, above.
- **GAE ended every episode one step late**, from 48042e5 (April) until #86
  (2026-09-27), for every algorithm on `GAEBuffer`. Experiments whose runs
  predate the fix ran with it; each findings file names the code it ran on.
  What the fix changes in their results has not been measured.
- **Some frames each buffer trained on came from the previous policy.** The
  server infers for all its clients at once, so when a buffer fills, some
  actions are still in flight. Their feedback went into the next buffer with
  the old policy's log-probability, or was dropped but counted, depending on
  timing. In the pi0.5 runs that was up to 9 of each buffer's 4,096 frames.
  It showed up as checkpoint names past `train_itrs × buffer_size` (E42's
  4100, 8200, 12290...), and #104 fixed it.

Four claims on an earlier version of this page were themselves wrong, and
were corrected rather than rewritten away:
- The E5 round-trip spread came from three observations, not five.
- The 1.76x was four runs against five, not five against five.
- E1's coexistence claim was disproved by three rounds, not four.
- Only E1's scripts pin their dependencies' SHAs, not all of them.

## What is missing

- **A cross-machine number on a fast link.** E43 crossed campus Wi-Fi. A
  wired LAN or a datacenter link is unmeasured; the cost model says how it
  scales.
- **RL that improves pi0.5.** E42 holds it; nothing here has improved it.
- **[E3](e3-integration-cost/) was never run.** It compares what adding an environment or a policy
  costs here and elsewhere. Its protocol is registered, including the
  familiarity bias that would favour PlugRL, and no data exists.

## Reproducing

Most scripts assume Linux. Several were driven from Windows through WSL, and
the `wsl-*.sh` wrappers set up a proxy and a clean git config before handing
off; adapt or ignore those. E6, E16, E19 and E20 ran on Windows with no
GPU.

E1 is the only experiment that installs from upstream forks, and its scripts
pin those forks' SHAs as resolved on the day. The rest pin less. E2 installs
the two local checkouts plus PyPI version ranges, and E5-E8 run against a
`.venv` that already exists, so rerunning those reproduces a working tree and
a range, not an exact stack. Where a findings file names the commit it ran
on, that is the code to rerun.
