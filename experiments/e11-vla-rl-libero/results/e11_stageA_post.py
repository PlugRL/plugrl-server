"""Turn one Stage A cell's raw output into rows, per PROTOCOL.md's record format.

Success rate per task comes from each env client process's summary.json. The
recorder appends every finished episode to a success window, whether or not it
writes per-episode artefacts, and the window holds 100 episodes by default. With
at most 20 episodes per task the window covers them all, so mean_success_rate
is exactly successes / episodes. That is checked, not assumed.

Correctness is reconciled against the server's own record. The eval algorithm
logs every episode's metrics at step 0, so the server's step counter says
nothing about how many environment steps it saw. What it does record is one
episode/length per finished episode. Every env client stops on an episode
boundary, so when the run is clean:

  server episodes            == episodes the clients report
  sum of server episode/length == env steps the clients report

The first smoke run of this harness is why both are checked: env client workers
left over from an earlier, killed run joined its server and ran ten episodes the
clients of the cell never saw. Only the server's record could show that.

PROTOCOL.md makes a row invalid when its step accounting disagrees, so every row
of a cell whose accounting is false - or cannot be established - is invalid.
"""

import argparse
import json
import math
import pathlib
import re

Z95 = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + Z95 * Z95 / n
    centre = p + Z95 * Z95 / (2 * n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def server_episode_record(ck_dir):
    """Episode count and summed length from the server's TensorBoard sink."""
    episodes = None
    length_sum = None
    tags_seen = set()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        for f in sorted(ck_dir.rglob("events.out.tfevents.*")):
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            tags_seen.update(tags)
            if "episode/success" in tags:
                episodes = (episodes or 0) + len(ea.Scalars("episode/success"))
            if "episode/length" in tags:
                length_sum = (length_sum or 0) + sum(int(round(e.value)) for e in ea.Scalars("episode/length"))
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        print("tensorboard read failed:", type(e).__name__, str(e)[:120])
    print("server scalar tags:", sorted(tags_seen)[:24])
    return episodes, length_sum


def verdict(server_value, client_value):
    if server_value is None:
        return "unavailable"
    return str(server_value == client_value).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--nep", type=int, required=True)
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--client-exit", type=int, required=True)
    ap.add_argument("--wall", type=int, required=True)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    res = pathlib.Path(a.res)
    rollout = out / "runs" / a.cell / "rollout"

    per_proc = {}
    for s in sorted(rollout.glob("proc_*/summary.json")):
        j = json.loads(s.read_text())
        per_proc[int(j["process_id"])] = j

    client_log = (out / "client.log").read_text(errors="replace") if (out / "client.log").exists() else ""
    server_log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""

    client_steps = sum(int(m) for m in re.findall(r"Final rollout timing summary: env_steps=(\d+)", client_log))
    finals = len(re.findall(r"Final rollout timing summary", client_log))
    warnings = server_log.count("arrived with no step state")
    reconnects = max(0, client_log.count("Waiting for server") - a.nproc)

    client_episodes = sum(int(j["completed_episodes"]) for j in per_proc.values())
    server_episodes, server_length_sum = server_episode_record(out / "ck")
    episode_accounting = verdict(server_episodes, client_episodes)
    step_accounting = verdict(server_length_sum, client_steps)
    accounting_ok = episode_accounting == "true" and step_accounting == "true"

    rows = []
    total_k = total_n = 0
    all_valid = a.client_exit == 0 and len(per_proc) == a.nproc and accounting_ok
    for pid in range(a.nproc):
        task = pid % 10
        j = per_proc.get(pid)
        if j is None:
            rows.append([a.cell, a.suite, task, 0, 0, "", "", "", "false"])
            all_valid = False
            continue
        n = int(j["completed_episodes"])
        window = int(j.get("metric_window", 100))
        k = int(round(float(j["mean_success_rate"]) * n))
        valid = a.client_exit == 0 and n == a.nep and window >= n and accounting_ok
        all_valid = all_valid and valid
        lo, hi = wilson(k, n)
        total_k += k
        total_n += n
        rows.append([a.cell, a.suite, task, n, k, f"{k / n:.4f}" if n else "", f"{lo:.4f}", f"{hi:.4f}", str(valid).lower()])

    header = "cell\tsuite\ttask_id\tepisodes\tsuccesses\tsr\tci_lo\tci_hi\tvalid\n"
    path = res / "stageA.tsv"
    new = not path.exists()
    with path.open("a") as fh:
        if new:
            fh.write(header)
        for r in rows:
            fh.write("\t".join(str(c) for c in r) + "\n")

    lo, hi = wilson(total_k, total_n)
    cpath = res / "stageA_correctness.tsv"
    cnew = not cpath.exists()
    with cpath.open("a") as fh:
        if cnew:
            fh.write("cell\tclient_exit\twall_s\tfinal_summaries\tclient_episodes\tclient_steps\t"
                     "server_episodes\tserver_length_sum\tepisode_accounting\tstep_accounting\t"
                     "no_step_state_warnings\treconnects\tall_valid\n")
        fh.write("\t".join(str(c) for c in [
            a.cell, a.client_exit, a.wall, finals, client_episodes, client_steps, server_episodes,
            server_length_sum, episode_accounting, step_accounting, warnings, reconnects,
            str(all_valid).lower()]) + "\n")

    print(f"CELL {a.cell} | {a.suite} | episodes {total_n} | successes {total_k} | "
          f"mean sr {total_k / total_n if total_n else float('nan'):.4f} [{lo:.4f}, {hi:.4f}] | valid {all_valid}")
    print(f"correctness: warnings {warnings} | reconnects {reconnects} | "
          f"episodes client {client_episodes} server {server_episodes} ({episode_accounting}) | "
          f"steps client {client_steps} server {server_length_sum} ({step_accounting})")
    for r in rows:
        print("   ", "\t".join(str(c) for c in r))


if __name__ == "__main__":
    main()
