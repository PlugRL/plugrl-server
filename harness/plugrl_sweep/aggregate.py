"""Turn a pile of per-cell results into the two tables the paper needs.

The compatibility matrix says which (environment, policy) combinations run.
The timing table says where their wall clock goes. Both are aggregations over
seeds, and both have to keep "we did not try this here" visibly distinct from
"we tried and it failed" - a matrix that blurs those is worse than no matrix.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from plugrl_sweep.outcome import CellResult, Outcome

#: How a combination reads in the matrix, once its seeds are folded together.
MARK = {
    Outcome.OK: "ok",
    Outcome.FAILED: "fail",
    Outcome.TIMEOUT: "timeout",
    Outcome.INCOMPLETE: "partial",
    Outcome.UNAVAILABLE: "-",
}


def load_results(path: Path) -> list[CellResult]:
    """Read a results.jsonl written by the sweep."""
    results = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(CellResult.from_dict(json.loads(line)))
    return results


def combine_seeds(outcomes: Sequence[Outcome]) -> Outcome:
    """Fold one combination's seeds into a single verdict.

    A combination counts as runnable only if every seed ran. One seed failing
    out of three is a fact about the combination, not noise to average away -
    so the worst outcome wins, except that UNAVAILABLE only survives if every
    seed was unavailable (otherwise some seeds clearly could be attempted).
    """
    if not outcomes:
        return Outcome.UNAVAILABLE
    if all(o is Outcome.UNAVAILABLE for o in outcomes):
        return Outcome.UNAVAILABLE
    evidence = [o for o in outcomes if o.is_evidence]
    for worse in (Outcome.FAILED, Outcome.TIMEOUT, Outcome.INCOMPLETE):
        if worse in evidence:
            return worse
    return Outcome.OK


def compatibility_matrix(
    results: Iterable[CellResult],
) -> tuple[list[str], list[str], dict[tuple[str, str], Outcome]]:
    by_combo: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for r in results:
        by_combo[(r.env_uid, r.policy_uid)].append(r.outcome)

    envs = sorted({env for env, _ in by_combo})
    policies = sorted({policy for _, policy in by_combo})
    cells = {combo: combine_seeds(outcomes) for combo, outcomes in by_combo.items()}
    return envs, policies, cells


def render_matrix(results: Iterable[CellResult]) -> str:
    results = list(results)
    envs, policies, cells = compatibility_matrix(results)
    if not envs:
        return "(no results)"

    width = max([len(e) for e in envs] + [3])
    col = max([len(p) for p in policies] + [8])

    lines = [
        "| "
        + "env".ljust(width)
        + " | "
        + " | ".join(p.ljust(col) for p in policies)
        + " |"
    ]
    lines.append(
        "|"
        + "-" * (width + 2)
        + "|"
        + "|".join("-" * (col + 2) for _ in policies)
        + "|"
    )
    for env in envs:
        row = [
            MARK.get(cells.get((env, p), Outcome.UNAVAILABLE), "?").ljust(col)
            for p in policies
        ]
        lines.append("| " + env.ljust(width) + " | " + " | ".join(row) + " |")

    counts: dict[Outcome, int] = defaultdict(int)
    for outcome in cells.values():
        counts[outcome] += 1
    summary = ", ".join(f"{MARK[o]}={counts[o]}" for o in Outcome if counts.get(o))
    lines.append("")
    lines.append(f"{len(cells)} combinations: {summary}")
    return "\n".join(lines)


def timing_table(results: Iterable[CellResult]) -> list[dict[str, Any]]:
    """Median stage fractions per combination, over the seeds that ran.

    Only OK runs contribute. A failed run's partial timings describe how far
    it got, not how the combination behaves.
    """
    by_combo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.outcome is Outcome.OK and r.timing:
            by_combo[(r.env_uid, r.policy_uid)].append(r.timing)

    rows = []
    for (env, policy), timings in sorted(by_combo.items()):
        row: dict[str, Any] = {"env": env, "policy": policy, "seeds": len(timings)}
        for field in (
            "effective_fps",
            "env_step_frac",
            "infer_wait_frac",
            "infer_obs_pack_frac",
            "feedback_frac",
        ):
            values = [t[field] for t in timings if field in t]
            row[field] = statistics.median(values) if values else None
        rows.append(row)
    return rows


def render_timing(results: Iterable[CellResult]) -> str:
    rows = timing_table(results)
    if not rows:
        return "(no completed runs to time)"

    header = f"{'env':<16} {'policy':<18} {'n':>2} {'fps':>8} {'env':>7} {'infer':>7} {'pack':>7} {'fb':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:

        def pct(key: str) -> str:
            v = r.get(key)
            return "-" if v is None else f"{v * 100:6.1f}%"

        fps = r.get("effective_fps")
        lines.append(
            f"{r['env']:<16} {r['policy']:<18} {r['seeds']:>2} "
            f"{'-' if fps is None else f'{fps:8.1f}'} "
            f"{pct('env_step_frac')} {pct('infer_wait_frac')} "
            f"{pct('infer_obs_pack_frac')} {pct('feedback_frac')}"
        )
    lines.append("")
    lines.append("env/infer/pack/fb are shares of collection wall clock.")
    return "\n".join(lines)
