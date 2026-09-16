"""Turn the Stage C training run into rows, per PROTOCOL.md's record format.

  results/stageC_train.tsv  step episodes rolling_sr mean_reward learn_s gpu_mem_gb valid

One row per learn iteration. Everything comes from the server's own TensorBoard
record, except peak memory, which nothing in the server measures and the
harness samples from outside every 15 s.

**The episode count is a lower bound, not a count.** The server logs the mean
success over the episodes that finished during an iteration, but never their
number: EpisodeMetricWindow keeps no count and is reset after each learn. The
mean of n booleans is k/n, so the count is a denominator of that fraction -
but 12/19, 24/38 and 36/57 are the same number, and only the smallest is
recoverable. The `episodes` column therefore holds that smallest denominator,
and `note` carries what the collected steps suggest the true multiple is. The
run's total, read from the env clients' own summaries once they exit, goes in
the correctness file, where it constrains those multiples.

**valid** reflects the run, not this bookkeeping: an iteration is valid if its
global step advanced by a full buffer and the server logged no
missing-step-state warning. The warnings cannot be attributed to one
iteration, so one of them invalidates every row, which is the conservative
reading of PROTOCOL.md.
"""

import argparse
import json
import pathlib
import re
from fractions import Fraction

TOLERANCE = 1e-6


def scalars(events_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(events_path), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return {tag: {e.step: (e.value, e.wall_time) for e in ea.Scalars(tag)} for tag in tags}


def episode_lower_bound(success_mean, reward_mean):
    """Smallest denominator of the logged mean, checked against the mean return."""
    if success_mean is None:
        return None, "no success metric"
    fraction = Fraction(float(success_mean)).limit_denominator(500)
    n = fraction.denominator
    if abs(float(fraction) - float(success_mean)) > TOLERANCE:
        return None, "mean is not a small fraction"
    if reward_mean is not None:
        scaled = float(reward_mean) * n
        if abs(scaled - round(scaled)) > 1e-4:
            return None, "the same denominator does not divide the mean return"
    return n, ""


def peak_memory_gb(samples, start_wall, end_wall):
    window = [mib for ts, mib in samples if start_wall <= ts <= end_wall]
    if not window:
        return None
    return max(window) / 1024.0


def client_total_episodes(out, cell):
    total = 0
    found = 0
    for path in sorted((out / "runs" / cell / "rollout").glob("proc_*/summary.json")):
        try:
            total += int(json.loads(path.read_text())["completed_episodes"])
            found += 1
        except Exception:  # noqa: BLE001 - a summary appears only when its client exits
            continue
    return (total, found) if found else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="C_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--buffer-size", type=int, default=4096)
    ap.add_argument("--replan-steps", type=int, default=5)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    res = pathlib.Path(a.res)

    events = sorted((out / "ck").rglob("events.out.tfevents.*"))
    if not events:
        raise SystemExit(f"no tensorboard events under {out / 'ck'}")
    data = scalars(events[0])

    server_log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""
    client_log = (out / "client.log").read_text(errors="replace") if (out / "client.log").exists() else ""
    warnings = server_log.count("arrived with no step state")
    feedback_timeouts = server_log.count("Timed out waiting for feedback")
    reconnects = client_log.count("Retrying in 5 seconds")
    client_feedback = sum(
        int(m) for m in re.findall(r"Final rollout timing summary:.*?feedback_calls=(\d+)", client_log)
    )
    client_steps = sum(
        int(m) for m in re.findall(r"Final rollout timing summary: env_steps=(\d+)", client_log)
    )
    total_episodes, summaries = client_total_episodes(out, a.cell)

    samples = []
    mem_csv = out / "gpu0_mem.csv"
    if mem_csv.exists():
        for line in mem_csv.read_text(errors="replace").splitlines():
            parts = line.split(",")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                samples.append((float(parts[0]), int(parts[1])))

    itrs = data.get("train/train_itrs", {})
    rows = []
    previous_wall = None
    previous_step = 0
    for step in sorted(itrs):
        _, wall = itrs[step]
        success = data.get("rollout/success", {}).get(step, (None, None))[0]
        reward = data.get("rollout/reward", {}).get(step, (None, None))[0]
        length = data.get("rollout/length", {}).get(step, (None, None))[0]
        learn_s = data.get("progress/learn_time", {}).get(step, (None, None))[0]
        collect_s = data.get("progress/collection_time", {}).get(step, (None, None))[0]
        step_delta = step - previous_step
        env_steps = step_delta * a.replan_steps

        bound, why = episode_lower_bound(success, reward)
        if bound is not None and length:
            ceiling = env_steps / float(length)
            multiples = [m for m in range(1, 9) if m * bound <= ceiling * 1.05]
            why = (
                f"lower bound; collected {env_steps} steps at mean length {length:.0f} "
                f"allows at most ~{ceiling:.0f} episodes, so {' or '.join(str(m * bound) for m in multiples) or bound}"
            )
        gpu_gb = peak_memory_gb(samples, previous_wall or (wall - (learn_s or 0) - (collect_s or 0)), wall)
        valid = step_delta == a.buffer_size and warnings == 0
        rows.append(dict(
            step=step,
            episodes=bound if bound is not None else "unavailable",
            rolling_sr=f"{success:.4f}" if success is not None else "",
            mean_reward=f"{reward:.4f}" if reward is not None else "",
            learn_s=f"{learn_s:.0f}" if learn_s is not None else "",
            gpu_mem_gb=f"{gpu_gb:.2f}" if gpu_gb is not None else "unavailable",
            valid=str(valid).lower(),
            collect_s=f"{collect_s:.0f}" if collect_s is not None else "",
            mean_length=f"{length:.1f}" if length is not None else "",
            note=why,
        ))
        previous_wall = wall
        previous_step = step

    columns = ["step", "episodes", "rolling_sr", "mean_reward", "learn_s", "gpu_mem_gb", "valid",
               "collect_s", "mean_length", "note"]
    path = res / "stageC_train.tsv"
    new = not path.exists()
    with path.open("a") as fh:
        if new:
            fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in columns) + "\n")

    server_step = max(itrs) if itrs else 0
    accounting = "true" if client_feedback and client_feedback == server_step else (
        "unavailable" if not client_feedback else "false"
    )
    cpath = res / "stageC_train_correctness.tsv"
    cnew = not cpath.exists()
    with cpath.open("a") as fh:
        if cnew:
            fh.write("cell\titerations\tserver_global_step\tclient_feedback_calls\tclient_env_steps\t"
                     "client_episodes\tclient_summaries\tstep_accounting\tno_step_state_warnings\t"
                     "feedback_timeouts\treconnects\n")
        fh.write("\t".join(str(c) for c in [
            a.cell, len(rows), server_step, client_feedback, client_steps,
            total_episodes if total_episodes is not None else "unavailable", summaries,
            accounting, warnings, feedback_timeouts, reconnects]) + "\n")

    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row[c]) for c in columns))
    print(f"\niterations {len(rows)} | server global_step {server_step} | client feedback_calls "
          f"{client_feedback} | step accounting {accounting}")
    print(f"client episodes {total_episodes} from {summaries} summaries | "
          f"missing-step-state warnings {warnings} | feedback timeouts {feedback_timeouts} | "
          f"client reconnect attempts {reconnects}")


if __name__ == "__main__":
    main()
