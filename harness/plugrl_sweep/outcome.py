"""What happened to a cell, and how sure we are.

The distinctions here are the whole point. A sweep that reports "failed" for
both "this environment needs an extra we did not install" and "this
environment raises on startup" produces a matrix nobody can act on - and one
that reports success on a run that never completed produces numbers nobody
should trust. Both mistakes were made by hand-written harnesses earlier in
this project, so they get named types here.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any


class Outcome(enum.Enum):
    #: Ran to completion and produced the episodes it was asked for.
    OK = "ok"
    #: Could not be attempted: an optional dependency is missing. Says nothing
    #: about whether the combination works.
    UNAVAILABLE = "unavailable"
    #: Started and failed. This is a real result about the combination.
    FAILED = "failed"
    #: Exceeded its time budget. Distinct from FAILED: it may simply be slow.
    TIMEOUT = "timeout"
    #: Exited zero without producing the expected episodes. Treated as a
    #: failure, but named separately because it usually means the harness
    #: asked the wrong question rather than the code being broken.
    INCOMPLETE = "incomplete"

    @property
    def counts_as_runnable(self) -> bool:
        return self is Outcome.OK

    @property
    def is_evidence(self) -> bool:
        """Whether this says anything about the combination itself."""
        return self is not Outcome.UNAVAILABLE


#: Exception types that mean "this machine is missing something", not "this
#: combination does not work".
_DEPENDENCY_ERRORS = (
    "DependencyNotInstalled",
    "ModuleNotFoundError",
)

#: Matches the last line of a Python traceback: "SomeError: some message".
_EXCEPTION_LINE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|NotInstalled)): ?(.*)$"
)


def final_exception(stdout: str) -> tuple[str, str] | None:
    """The exception a traceback ended on, if it ended on one.

    Reading the *last* exception rather than searching the whole log is the
    point. The env client warns about every extra it could not import on every
    run, so a log always mentions several packages being absent no matter
    which env was asked for.
    """
    for line in reversed(stdout.splitlines()):
        match = _EXCEPTION_LINE.match(line.strip())
        if match:
            return match.group(1), match.group(2)
    return None


def looks_unavailable(stdout: str, env_uid: str) -> bool:
    """Did this cell fail because the machine lacks something, rather than
    because the combination is broken?

    Two shapes count. Either the env never registered, so tyro rejected the
    name outright:

        Argument {dummy-v1,atari-v1}: invalid choice: 'libero-v1'

    Or it registered and then could not be constructed because an optional
    dependency is absent - gymnasium raises DependencyNotInstalled for this,
    and the env wrappers here raise ImportError with a message naming the
    extra to install.

    Anything else that registered and then failed is a real result about the
    combination, and belongs in the failure column.
    """
    if f"invalid choice: '{env_uid}'" in stdout:
        return True

    exception = final_exception(stdout)
    if exception is None:
        return False
    kind, message = exception
    # Tracebacks print the qualified name - gymnasium.error.DependencyNotInstalled,
    # not DependencyNotInstalled - so compare on the final component.
    short = kind.rsplit(".", 1)[-1]
    if short in _DEPENDENCY_ERRORS:
        return True
    return short == "ImportError" and "not installed" in message


def classify(
    *,
    returncode: int,
    stdout: str,
    episodes_expected: int,
    episodes_completed: int | None,
    env_uid: str = "",
    timed_out: bool = False,
) -> Outcome:
    if timed_out:
        return Outcome.TIMEOUT
    if returncode != 0:
        if env_uid and looks_unavailable(stdout, env_uid):
            return Outcome.UNAVAILABLE
        return Outcome.FAILED
    if episodes_completed is None or episodes_completed < episodes_expected:
        return Outcome.INCOMPLETE
    return Outcome.OK


@dataclasses.dataclass
class CellResult:
    key: str
    env_uid: str
    policy_uid: str
    algo_uid: str
    seed: int
    outcome: Outcome
    duration_s: float = 0.0
    episodes_completed: int | None = None
    detail: str = ""
    timing: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CellResult:
        d = dict(d)
        d["outcome"] = Outcome(d["outcome"])
        return cls(**d)
