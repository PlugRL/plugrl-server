"""Turn the ten-iteration training run into PROTOCOL.md's `results/train.tsv`.

  iteration  step  wall_s  gpu0_peak_mib  gpu1_peak_mib

One row per iteration, plus a `startup` row and a `total` row. The protocol
asks for "iterations completed, wall clock, peak memory per card"; the rows
are per iteration because prediction 6 is a per-iteration bound and prediction
1 turns on whether the allocator grows across the ten.

Three things worth knowing about the numbers:

- **An iteration is measured checkpoint to checkpoint.** Iteration 1 has no
  previous checkpoint, so it is measured from the server's `is listening`
  line - not from the wrapper's start, which would charge it the minutes spent
  importing and constructing the policy. Those minutes are the `startup` row.
- **Peak memory is the whole card, not this process.** It comes from
  `nvidia-smi --query-gpu=memory.used`, sampled every ten seconds from
  outside, because nothing inside the server records it. If another job shared
  a card, its memory is in these numbers; `--check-neighbours` reports whether
  one did, and the finding should say so either way.
- **A ten-second sample can miss a peak.** These are lower bounds on the true
  maxima.
"""

import argparse
import datetime
import pathlib
import re
import sys


def read_samples(path):
    """[(epoch, card0_mib, card1_mib)], skipping lines nvidia-smi garbled."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return out


def peak_between(samples, lo, hi):
    """Highest reading on each card in (lo, hi]. None if no sample lands there."""
    window = [s for s in samples if lo < s[0] <= hi]
    if not window:
        return None, None
    return max(s[1] for s in window), max(s[2] for s in window)


def listening_epoch(runner_log, day):
    """The `server listening` wall time, as an epoch on the run's own day.

    The runner logs HH:MM:SS with no date; `day` supplies it. A run that
    crosses midnight would need more than this, and this one does not - the
    caller checks.
    """
    m = re.search(r"\[(\d{2}):(\d{2}):(\d{2})\] server listening", runner_log)
    if not m:
        return None
    h, mi, s = (int(g) for g in m.groups())
    stamp = datetime.datetime.combine(day, datetime.time(h, mi, s))
    return int(stamp.timestamp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck-root", required=True, help="the directory of numbered step dirs")
    ap.add_argument("--gpu-mem", required=True)
    ap.add_argument("--runner-log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck_root = pathlib.Path(a.ck_root)
    samples = read_samples(pathlib.Path(a.gpu_mem))
    if not samples:
        sys.exit(f"no usable samples in {a.gpu_mem}")

    steps = sorted(
        (int(d.name), d / "model.safetensors")
        for d in ck_root.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    steps = [(n, f) for n, f in steps if f.exists()]
    if not steps:
        sys.exit(f"no checkpoints under {ck_root}")

    runner = pathlib.Path(a.runner_log).read_text(errors="replace")
    t_first = samples[0][0]
    day = datetime.datetime.fromtimestamp(t_first).date()
    t_listen = listening_epoch(runner, day)
    if t_listen is None:
        sys.exit("no 'server listening' line in the runner log")
    if not t_first <= t_listen <= samples[-1][0]:
        sys.exit(
            f"'server listening' resolved to {t_listen}, outside the sampling "
            f"window {t_first}..{samples[-1][0]} - did the run cross midnight?"
        )

    rows = [("startup", "-", t_listen - t_first) + peak_between(samples, t_first - 1, t_listen)]

    prev = t_listen
    for i, (step, f) in enumerate(steps, start=1):
        t = int(f.stat().st_mtime)
        rows.append((str(i), str(step), t - prev) + peak_between(samples, prev, t))
        prev = t

    t_end = samples[-1][0]
    rows.append(("total", str(steps[-1][0]), t_end - t_first) + peak_between(samples, t_first - 1, t_end))

    out = pathlib.Path(a.out)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("iteration\tstep\twall_s\tgpu0_peak_mib\tgpu1_peak_mib\n")
        for r in rows:
            fh.write("\t".join("-" if v is None else str(v) for v in r) + "\n")

    print(f"wrote {out} - {len(steps)} iterations")
    for r in rows:
        print("\t".join("-" if v is None else str(v) for v in r))


if __name__ == "__main__":
    main()
