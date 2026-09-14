"""Turn one Stage C evaluation cell into a row, per PROTOCOL.md's record format.

  results/stageC_eval.tsv   policy task_id episodes successes sr ci_lo ci_hi valid

One env client process ran every episode, so there is one summary.json. Its
success window holds 100 episodes, and the protocol evaluates 50, so
mean_success_rate is exactly successes / episodes. That is checked, not
assumed.

Correctness is reconciled against the server's own record, as in Stage A: its
episode count against the client's, and the sum of its episode lengths against
the client's environment steps. PROTOCOL.md makes a row invalid when its step
accounting disagrees, and a row whose accounting cannot be established is
invalid too.
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e11_stageA_post import server_episode_record, verdict, wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--policy-label", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--nep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--client-exit", type=int, required=True)
    ap.add_argument("--wall", type=int, required=True)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    res = pathlib.Path(a.res)
    summaries = sorted((out / "runs" / a.cell / "rollout").glob("proc_*/summary.json"))

    client_log = (out / "client.log").read_text(errors="replace") if (out / "client.log").exists() else ""
    server_log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""
    client_steps = sum(int(m) for m in re.findall(r"Final rollout timing summary: env_steps=(\d+)", client_log))
    warnings = server_log.count("arrived with no step state")
    reconnects = max(0, client_log.count("Waiting for server") - 1)

    n = k = 0
    window = 0
    if len(summaries) == 1:
        j = json.loads(summaries[0].read_text())
        n = int(j["completed_episodes"])
        window = int(j.get("metric_window", 100))
        k = int(round(float(j["mean_success_rate"]) * n))

    server_episodes, server_length_sum = server_episode_record(out / "ck")
    episode_accounting = verdict(server_episodes, n)
    step_accounting = verdict(server_length_sum, client_steps)
    accounting_ok = episode_accounting == "true" and step_accounting == "true"

    valid = (
        a.client_exit == 0
        and len(summaries) == 1
        and n == a.nep
        and window >= n
        and accounting_ok
    )
    lo, hi = wilson(k, n)

    path = res / "stageC_eval.tsv"
    new = not path.exists()
    with path.open("a") as fh:
        if new:
            fh.write("policy\ttask_id\tepisodes\tsuccesses\tsr\tci_lo\tci_hi\tvalid\n")
        fh.write("\t".join(str(c) for c in [
            a.policy_label, a.task_id, n, k, f"{k / n:.4f}" if n else "",
            f"{lo:.4f}" if n else "", f"{hi:.4f}" if n else "", str(valid).lower()]) + "\n")

    cpath = res / "stageC_eval_correctness.tsv"
    cnew = not cpath.exists()
    with cpath.open("a") as fh:
        if cnew:
            fh.write("cell\tpolicy\tclient_exit\twall_s\tclient_episodes\tclient_steps\tserver_episodes\t"
                     "server_length_sum\tepisode_accounting\tstep_accounting\tno_step_state_warnings\t"
                     "reconnects\tvalid\n")
        fh.write("\t".join(str(c) for c in [
            a.cell, a.policy_label, a.client_exit, a.wall, n, client_steps, server_episodes,
            server_length_sum, episode_accounting, step_accounting, warnings, reconnects,
            str(valid).lower()]) + "\n")

    print(f"CELL {a.cell} | {a.policy_label} | task {a.task_id} | episodes {n} | successes {k} | "
          f"sr {k / n if n else float('nan'):.4f} [{lo:.4f}, {hi:.4f}] | valid {valid}")
    print(f"correctness: warnings {warnings} | reconnects {reconnects} | "
          f"episodes client {n} server {server_episodes} ({episode_accounting}) | "
          f"steps client {client_steps} server {server_length_sum} ({step_accounting})")


if __name__ == "__main__":
    main()
