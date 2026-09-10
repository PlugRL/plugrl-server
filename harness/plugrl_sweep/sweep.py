"""Drive a matrix of cells and write what came back.

Results are appended as they finish, so an interrupted sweep leaves usable
data and a resumed one can skip what it already has. On a cluster that
matters more than it sounds: a sweep that has to complete or be thrown away
will be thrown away.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from plugrl_sweep.aggregate import load_results, render_matrix, render_timing
from plugrl_sweep.cell import Cell
from plugrl_sweep.outcome import CellResult, Outcome
from plugrl_sweep.runner import Executables, run_cell


def build_matrix(
    envs: Sequence[str],
    policies: Sequence[str],
    seeds: Sequence[int],
    *,
    num_episodes: int = 2,
    policy_overrides: dict | None = None,
    per_env_policy_overrides: dict[str, dict] | None = None,
    algo_overrides: dict | None = None,
) -> list[Cell]:
    """Cross-product of envs, policies and seeds.

    A policy has to match the environment's action space, and environments
    disagree about that: CartPole takes one discrete action, LIBERO takes a
    seven-dimensional continuous one. A single global override cannot serve
    both, so per_env_policy_overrides layers on top of policy_overrides for
    the envs that need it. Without this the matrix can only ever be one column
    wide.
    """
    per_env = per_env_policy_overrides or {}
    cells = []
    for env in envs:
        overrides = dict(policy_overrides or {})
        overrides.update(per_env.get(env, {}))
        for policy in policies:
            for seed in seeds:
                cells.append(
                    Cell(
                        env_uid=env,
                        policy_uid=policy,
                        seed=seed,
                        num_episodes=num_episodes,
                        policy_overrides=dict(overrides),
                        algo_overrides=dict(algo_overrides or {}),
                    )
                )
    return cells


def already_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    return {r.key for r in load_results(results_path)}


def run_sweep(
    cells: Iterable[Cell],
    *,
    executables: Executables,
    output_dir: Path,
    results_path: Path | None = None,
    resume: bool = True,
    client_timeout: float = 600.0,
) -> list[CellResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_path or (output_dir / "results.jsonl")

    done = already_done(results_path) if resume else set()
    if not resume and results_path.exists():
        results_path.unlink()

    cells = list(cells)
    pending = [c for c in cells if c.key not in done]
    print(
        f"{len(cells)} cells, {len(cells) - len(pending)} already done, {len(pending)} to run"
    )

    results: list[CellResult] = load_results(results_path) if done else []
    for index, cell in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {cell.key} ... ", end="", flush=True)
        started = time.monotonic()
        result = run_cell(
            cell,
            executables=executables,
            output_dir=output_dir,
            client_timeout=client_timeout,
        )
        print(f"{result.outcome.value} ({time.monotonic() - started:.0f}s)")
        if result.detail and result.outcome is not Outcome.OK:
            print(f"      {result.detail[:160]}")

        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict()) + "\n")
        results.append(result)

    return results


def _resolve(executable: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which(executable)
    if found is None:
        raise SystemExit(
            f"{executable} is not on PATH. Pass --{executable.replace('-', '')}-bin, "
            "or activate the environment that provides it."
        )
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("sweep-out"))
    parser.add_argument("--server-bin", default=None)
    parser.add_argument("--client-bin", default=None)
    parser.add_argument("--client-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="discard previous results instead of skipping finished cells",
    )
    parser.add_argument(
        "--policy-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="passed to the server as --policy.KEY VALUE (repeatable)",
    )
    parser.add_argument(
        "--env-policy-override",
        action="append",
        default=[],
        metavar="ENV:KEY=VALUE",
        help=(
            "policy override for one env only, e.g. classic-v1:discrete=true. "
            "Needed because a policy must match the env's action space and "
            "environments disagree about that. Repeatable."
        ),
    )
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.policy_override:
        key, _, value = item.partition("=")
        overrides[key] = _coerce(value)

    per_env: dict[str, dict] = {}
    for item in args.env_policy_override:
        env, _, rest = item.partition(":")
        key, _, value = rest.partition("=")
        if not env or not key:
            raise SystemExit(
                f"--env-policy-override expects ENV:KEY=VALUE, got {item!r}"
            )
        per_env.setdefault(env, {})[key] = _coerce(value)

    cells = build_matrix(
        args.envs,
        args.policies,
        args.seeds,
        num_episodes=args.episodes,
        policy_overrides=overrides,
        per_env_policy_overrides=per_env,
    )
    executables = Executables(
        server=_resolve("plugrl-run-server", args.server_bin),
        client=_resolve("plugrl-run-env-client", args.client_bin),
    )

    results = run_sweep(
        cells,
        executables=executables,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        client_timeout=args.client_timeout,
    )

    print()
    print(render_matrix(results))
    print()
    print(render_timing(results))
    return 0


def _coerce(value: str):
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


if __name__ == "__main__":
    sys.exit(main())
