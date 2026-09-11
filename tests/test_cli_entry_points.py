"""Every advertised entry point has to at least start.

Two ways this package has shipped commands that could not run, both of them
invisible to a test suite that only imported library code:

  * `cli.py` and `cli_ray.py` had no `if __name__ == "__main__"` guard, so
    the eleven `python -m plugrl_server.cli ...` invocations in the README
    printed nothing and exited 0;
  * `cli_ray.py` looked for `DDPAlgorithm` on `base_algorithm`, where it has
    never been - it lives in `algorithm.distributed`, which is where the
    other six modules that use it import it from - so
    `plugrl-run-server-ray` died with AttributeError.

Neither needs a GPU, a cluster or a training run to catch.
"""

import subprocess
import sys

import pytest

MODULES = ["plugrl_server.cli", "plugrl_server.cli_ray"]


@pytest.mark.parametrize("module", MODULES)
def test_python_m_actually_runs_it(module):
    """Without a __main__ guard this exits 0 having done nothing at all."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    combined = result.stdout + result.stderr
    assert "usage:" in combined.lower(), (
        f"`python -m {module} --help` produced no usage text; it most likely "
        f"has no __main__ guard. stdout={result.stdout[:400]!r}"
    )


@pytest.mark.parametrize("module", MODULES)
def test_the_module_imports(module):
    """An entry point that cannot be imported cannot be run."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_cli_ray_resolves_the_class_it_type_checks_against():
    """The name has to resolve where cli_ray looks for it, not just exist."""
    from plugrl_server.algorithm.distributed import DDPAlgorithm
    from plugrl_server import cli_ray

    assert cli_ray.DDPAlgorithm is DDPAlgorithm
