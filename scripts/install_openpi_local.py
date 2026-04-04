from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _python_has_module(python_exe: str, module_name: str) -> bool:
    result = subprocess.run(
        [
            python_exe,
            "-c",
            (
                "import importlib.util, sys; "
                f"raise SystemExit(0 if importlib.util.find_spec('{module_name}') else 1)"
            ),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _build_install_cmd(
    python_exe: str,
    project_dir: pathlib.Path,
    *,
    skip_deps: bool,
) -> list[str]:
    uv_exe = shutil.which("uv")
    if uv_exe is not None:
        cmd = [uv_exe, "pip", "install", "--python", python_exe, "-e", str(project_dir)]
        if skip_deps:
            cmd.append("--no-deps")
        return cmd

    if _python_has_module(python_exe, "pip"):
        cmd = [python_exe, "-m", "pip", "install", "-e", str(project_dir)]
        if skip_deps:
            cmd.append("--no-deps")
        return cmd

    raise RuntimeError(
        "Could not find an installer for editable packages. "
        "Neither `uv` is available on PATH nor `pip` is installed in the target Python environment."
    )


def _install_editable(
    python_exe: str, project_dir: pathlib.Path, *, skip_deps: bool
) -> None:
    cmd = _build_install_cmd(python_exe, project_dir, skip_deps=skip_deps)
    _run(cmd)


def _patch_transformers(openpi_base_dir: pathlib.Path) -> pathlib.Path:
    import transformers

    transformers_dir = pathlib.Path(transformers.__file__).resolve().parent
    patch_dir = (
        openpi_base_dir / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    )
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"Missing transformers patch directory: {patch_dir}")

    shutil.copytree(patch_dir, transformers_dir, dirs_exist_ok=True)
    return transformers_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Editable-install local openpi-base into the current Python environment and apply the required transformers patch."
    )
    parser.add_argument(
        "--openpi-base-dir",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/openpi-base"),
        help="Path to the local openpi-base repository.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Install openpi-client/openpi in editable mode without resolving dependencies.",
    )
    args = parser.parse_args()

    openpi_base_dir = args.openpi_base_dir.resolve()
    openpi_client_dir = openpi_base_dir / "packages" / "openpi-client"

    if not openpi_base_dir.is_dir():
        raise FileNotFoundError(f"openpi-base directory does not exist: {openpi_base_dir}")
    if not openpi_client_dir.is_dir():
        raise FileNotFoundError(
            f"openpi-client package directory does not exist: {openpi_client_dir}"
        )

    python_exe = sys.executable
    print(f"Installing into Python environment: {python_exe}", flush=True)
    _install_editable(python_exe, openpi_client_dir, skip_deps=args.skip_deps)
    _install_editable(python_exe, openpi_base_dir, skip_deps=args.skip_deps)

    patched_dir = _patch_transformers(openpi_base_dir)
    print(f"Patched transformers package in: {patched_dir}", flush=True)
    print("Local openpi-base installation completed.", flush=True)


if __name__ == "__main__":
    main()
