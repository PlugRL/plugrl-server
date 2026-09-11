"""One cell of the evaluation matrix: an environment, a policy, a seed.

A cell is a description, not a run. It knows how to state its own identity and
how to turn itself into the two command lines that execute it, and nothing
else - so the matrix can be inspected, filtered and diffed without launching
anything.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Cell:
    env_uid: str
    policy_uid: str
    algo_uid: str = "dummy"
    seed: int = 0

    policy_variant: str = "default"
    algo_variant: str = "default"

    num_envs: int = 1
    num_episodes: int = 2

    # Flat "--policy.action-dim": 7 style overrides, kept separate so a cell
    # can be compared on identity without its tuning noise.
    policy_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    algo_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    env_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identity, used for filenames and for joining results."""
        return f"{self.env_uid}__{self.policy_uid}__{self.algo_uid}__s{self.seed}"

    @property
    def combo(self) -> tuple[str, str]:
        """The matrix coordinate, ignoring seed."""
        return (self.env_uid, self.policy_uid)

    def server_command(self, executable: str, port: int) -> list[str]:
        cmd = [
            executable,
            self.policy_uid,
            self.policy_variant,
            self.algo_uid,
            self.algo_variant,
            "--port",
            str(port),
            "--seed",
            str(self.seed),
        ]
        cmd += _flags("policy", self.policy_overrides)
        cmd += _flags("algo", self.algo_overrides)
        return cmd

    def client_command(self, executable: str, port: int, host: str) -> list[str]:
        cmd = [
            executable,
            self.env_uid,
            "--server-host",
            host,
            "--server-port",
            str(port),
            "--num-envs",
            str(self.num_envs),
            "--num-episodes",
            str(self.num_episodes),
            "--runner.seed",
            str(self.seed),
        ]
        cmd += _flags("env", self.env_overrides)
        return cmd

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _flags(prefix: str, overrides: dict[str, Any]) -> list[str]:
    """Render overrides as tyro flags.

    Booleans become --x/--no-x rather than a value, which is what tyro
    generates for a bool field; passing "True" would be read as a positional.
    """
    out: list[str] = []
    for key, value in overrides.items():
        name = key.replace("_", "-")
        if isinstance(value, bool):
            out.append(f"--{prefix}.{'' if value else 'no-'}{name}")
        else:
            out.extend([f"--{prefix}.{name}", str(value)])
    return out
