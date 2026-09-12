"""Every command in the README must survive its own argument parser.

An adversarial audit of this repository found six commands in README.md that
could not run: three `cli_ray` examples naming an algorithm the Ray entry point
rejects, `--track.enabled true` and `--resume true` written as though they took
a value when tyro renders them as bare flags, and a documented
`--algo.n_train_itr` that nothing reads. Every one had been in the file for as
long as the file had existed, and nothing would ever have caught them, because
a README is not executed.

This executes it, as far as the parser. A command that parses can still be
wrong - `n_train_itr` parsed perfectly and was ignored - so this is a floor,
not a guarantee. It is the floor that was missing.

Commands naming a policy or algorithm behind an optional extra are reported as
skipped, decided from the registries rather than from the error text, so a
genuine parse failure can never be mistaken for a missing extra.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import sys

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

ENTRY_POINTS = {
    "plugrl-run-server": "plugrl_server.cli",
    "plugrl-run-server-ray": "plugrl_server.cli_ray",
}


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:bash|sh|console)\n(.*?)```", text, re.S)


def _commands(text: str) -> list[str]:
    """Every server invocation in the README, line continuations joined."""
    out: list[str] = []
    for block in _bash_blocks(text):
        joined = re.sub(r"\\s*\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^(plugrl-run-server(-ray)?|python -m plugrl_server\.)", line):
                out.append(line)
    return out


def _module_and_args(command: str) -> tuple[str, list[str]]:
    parts = shlex.split(command)
    if parts[0] == "python":
        return parts[2], parts[3:]
    return ENTRY_POINTS[parts[0]], parts[1:]


def _load_cli(module_name: str):
    if module_name == "plugrl_server.cli":
        from plugrl_server.cli import cli
    elif module_name == "plugrl_server.cli_ray":
        from plugrl_server.cli_ray import cli
    else:  # pragma: no cover - the regex above admits nothing else
        raise AssertionError(module_name)
    return cli


COMMANDS = _commands(README.read_text(encoding="utf-8"))


def test_the_readme_contains_commands_to_check():
    """A regex that silently matched nothing would make every test below pass."""
    assert len(COMMANDS) >= 5, f"only found {len(COMMANDS)} commands in {README}"


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c[:60])
def test_documented_command_parses(command, monkeypatch, capsys):
    module_name, args = _module_and_args(command)
    if "--help" in args:
        pytest.skip("--help exits by design")

    from plugrl_server.algorithm.registration import REGISTERED_ALGO_CONFIGS
    from plugrl_server.policy.registration import REGISTERED_POLICY_CONFIGS

    positional: list[str] = []
    for token in args:
        if token.startswith("-"):
            break
        positional.append(token)
    if positional and positional[0] not in REGISTERED_POLICY_CONFIGS:
        pytest.skip(f"policy '{positional[0]}' needs an extra this install lacks")
    if len(positional) > 2 and positional[2] not in REGISTERED_ALGO_CONFIGS:
        pytest.skip(f"algorithm '{positional[2]}' needs an extra this install lacks")

    cli = _load_cli(module_name)
    monkeypatch.setattr(sys, "argv", ["prog", *args])

    try:
        cli()
    except SystemExit as exc:
        out = capsys.readouterr()
        tail = (out.out + out.err)[-1500:]
        pytest.fail(
            f"README command failed to parse (exit {exc.code}):\n  {command}\n\n{tail}"
        )
