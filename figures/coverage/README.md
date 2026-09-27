# The coverage figure

The figure on the project page's home page: every combination of the two MLP
policies and the two algorithms on four tasks, and pi0.5 on LIBERO. This
directory made its clips, stills and data; the page itself lives in
[plugrl.github.io](https://github.com/PlugRL/plugrl.github.io)
(`docs/media/coverage/`, `docs/javascripts/coverage.js`).

`cells.json` is the source of truth. For each cell it names the experiment,
the status that experiment established, a short note in English and Chinese,
and the run to take the clip from.

## How a clip is chosen

- **The seed** is the one whose mean return over its last ten iterations is
  the median of the three (ties go to the lowest seed).
- **The checkpoint** is that seed's final one.
- **The episode**: the checkpoint is evaluated for five episodes with the
  `eval` algorithm, which acts without training's sampling noise, on MuJoCo
  seed 100 (robomimic refuses a seed). The clip is the episode whose return
  is the median, not the best.
- **The timing**: real time, at most 25 fps, cut to eight seconds. An
  episode shorter than three seconds is slowed to fill three. LIBERO clips
  play the whole episode sped up to fit ten seconds. The page shows the
  factor whenever it is not about 1.
- **pi0.5** is recorded once per policy on the same scene: LIBERO-10 task 8,
  initial state 0, the first state the released policy solves in E26's
  fifty-episode evaluation. The page's numbers are those evaluations' (E25,
  E26), not the clip's.

The status of a cell is what the experiment found over three seeds. A clip
is one episode and can look better or worse than that.

## Reproducing

On `guangzhao`, from a checkout of this repository:

```bash
bash run.sh                 # record.py then pick.py for every MLP cell, about 3 minutes
python data.py              # coverage.json: statuses, notes, curves, ranges, experiment links
```

`record.py` runs under the server's venv, `pick.py` under the env client's
(it needs imageio and Pillow), `data.py` under the server's (it needs
tensorboard). The runs are found under `PLUGRL_HOME` (default
`~/zuogou/plugrl`); the three locally trained cells (E16, E18, E23) were
copied to `coverage/stage/` there first.

The pi0.5 clips are recorded on `qz103`:

```bash
bash libero_derive.sh       # E26's eval.sh with the recorder on, output under coverage/
bash libero_record.sh       # three one-episode evaluations at once
```

then copied to `media/raw/pi0-{base,fpo,dppo}/` and encoded with
`pick.py pi0-base` and so on.

Everything under `media/` stays out of git here. Copy
`media/site/*.{mp4,jpg}` and `coverage.json` to the page's
`docs/media/coverage/`.
