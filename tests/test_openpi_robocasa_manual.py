from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import time
import uuid

import pytest

REPO_ROOT = pathlib.Path("/mnt/plugrl/plugrl-server")
ENV_CLIENT_ROOT = pathlib.Path("/mnt/plugrl/plugrl-env-client")
OPENPI_BASE_ROOT = pathlib.Path("/mnt/openpi-base")

MANUAL_SERVER_PYTHON = pathlib.Path(
    os.environ.get("PLUGRL_SERVER_OPENPI_PYTHON", "/mnt/openpi-base/.venv/bin/python")
)
MANUAL_ENV_CLIENT_PYTHON = pathlib.Path(
    os.environ.get(
        "PLUGRL_ROBOCASA_ENV_CLIENT_PYTHON",
        "/mnt/plugrl/plugrl-env-client/.venv/bin/python",
    )
)
MANUAL_MAX_EPISODE_STEPS = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_MAX_EPISODE_STEPS", "40")
)
MANUAL_SERVER_WAIT_SECONDS = float(
    os.environ.get("PLUGRL_ROBOCASA_SERVER_WAIT_SECONDS", "300")
)
MANUAL_CLIENT_TIMEOUT_SECONDS = float(
    os.environ.get("PLUGRL_ROBOCASA_CLIENT_TIMEOUT_SECONDS", "180")
)
MANUAL_TASK_NAME = os.environ.get("OPENPI_ROBOCASA_TASK_NAME", "StoreLeftoversInBowl")
MANUAL_SPLIT = os.environ.get("OPENPI_ROBOCASA_SPLIT", "target")


def _server_support_site_packages() -> list[pathlib.Path]:
    env_root = REPO_ROOT / ".venv" / "lib"
    return sorted(env_root.glob("python*/site-packages"))


def _have_manual_inputs() -> bool:
    required = (
        "OPENPI_ROBOCASA_CONFIG_NAME",
        "OPENPI_ROBOCASA_CHECKPOINT_DIR",
        "OPENPI_ROBOCASA_DATASET_DIR",
    )
    return all(os.environ.get(key) for key in required)


def _server_python_has_dppo() -> bool:
    candidate_paths = [
        str(path) for path in _server_support_site_packages()
    ]
    pythonpath = os.pathsep.join([str(REPO_ROOT / "src"), *candidate_paths])
    result = subprocess.run(
        [
            str(MANUAL_SERVER_PYTHON),
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('dppo') else 1)",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
    return result.returncode == 0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _wait_for_port(
    host: str,
    port: int,
    *,
    timeout_s: float,
    process: subprocess.Popen[bytes] | None = None,
    log_path: pathlib.Path | None = None,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                log_text = _read_log(log_path) if log_path is not None else ""
                raise RuntimeError(
                    "Server process exited before opening the port. "
                    f"returncode={returncode} log={log_text}"
                )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
            except OSError:
                time.sleep(1.0)
                continue
            return
    raise TimeoutError(f"Timed out waiting for server on {host}:{port}")


def _build_server_env() -> dict[str, str]:
    pythonpath_entries = [
        str(REPO_ROOT / "src"),
        *[str(path) for path in _server_support_site_packages()],
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _build_client_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ENV_CLIENT_ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "WS_PROXY",
        "WSS_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ws_proxy",
        "wss_proxy",
    ):
        env.pop(key, None)
    return env


@pytest.mark.manual
@pytest.mark.skipif(
    not (
        _have_manual_inputs()
        and MANUAL_SERVER_PYTHON.exists()
        and MANUAL_ENV_CLIENT_PYTHON.exists()
        and _server_python_has_dppo()
    ),
    reason="Manual RoboCasa/OpenPI inputs, python environments, or DPPO dependency are not available.",
)
def test_manual_openpi_robocasa_dppo_roundtrip(tmp_path: pathlib.Path):
    port = _pick_free_port()
    run_suffix = uuid.uuid4().hex[:8]
    exp_name = f"manual-openpi-robocasa-{run_suffix}"
    checkpoint_base_dir = tmp_path / "checkpoints"
    policy_debug_dir = tmp_path / "policy_debug"
    server_log_path = tmp_path / "server.log"
    client_log_path = tmp_path / "client.log"

    server_log = server_log_path.open("wb")
    server = subprocess.Popen(
        [
            str(MANUAL_SERVER_PYTHON),
            "-u",
            "-m",
            "plugrl_server.cli",
            "pi0-policy",
            "robocasa",
            "dppo",
            "default",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log_level",
            "info",
            "--exp_name",
            exp_name,
            "--checkpoint_base_dir",
            str(checkpoint_base_dir),
            "--no-show_metric_table",
            "--no-show_progress_bar",
            "--algo.buffer_size",
            "1024",
            "--algo.train_itrs",
            "1",
            "--policy.name",
            os.environ["OPENPI_ROBOCASA_CONFIG_NAME"],
            "--policy.checkpoint_path",
            os.environ["OPENPI_ROBOCASA_CHECKPOINT_DIR"],
            "--policy.dataset_dir",
            os.environ["OPENPI_ROBOCASA_DATASET_DIR"],
            "--policy.export_debug_artifacts",
            "--policy.debug_artifact_dir",
            str(policy_debug_dir),
        ],
        cwd=REPO_ROOT,
        env=_build_server_env(),
        stdout=server_log,
        stderr=server_log,
    )

    try:
        try:
            _wait_for_port(
                "127.0.0.1",
                port,
                timeout_s=MANUAL_SERVER_WAIT_SECONDS,
                process=server,
                log_path=server_log_path,
            )
        except (TimeoutError, RuntimeError) as exc:
            raise type(exc)(
                f"{exc}. Server log: {_read_log(server_log_path)}"
            ) from exc

        client_log = client_log_path.open("wb")
        client_timed_out = False
        try:
            client = subprocess.run(
                [
                    str(MANUAL_ENV_CLIENT_PYTHON),
                    "-u",
                    "-m",
                    "plugrl_env_client.cli",
                    "robocasa-v1",
                    "--num-envs",
                    "1",
                    "--num-procs",
                    "1",
                    "--num-episodes",
                    "1",
                    "--exp-name",
                    exp_name,
                    "--server-host",
                    "127.0.0.1",
                    "--server-port",
                    str(port),
                    "--runner.max_episode_steps",
                    str(MANUAL_MAX_EPISODE_STEPS),
                    "--env.task_name",
                    MANUAL_TASK_NAME,
                    "--env.split",
                    MANUAL_SPLIT,
                    "--env.action_encoding",
                    "passthrough",
                    "--recorder.episode_freq",
                    "1",
                    "--recorder.no-thread0-only",
                    "--recorder.record_video",
                    "--recorder.record_full_rollout",
                    "--recorder.record_debug_packets",
                ],
                cwd=ENV_CLIENT_ROOT,
                env=_build_client_env(),
                stdout=client_log,
                stderr=client_log,
                check=False,
                timeout=MANUAL_CLIENT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            client = exc
            client_timed_out = True
        finally:
            client_log.close()

        if not client_timed_out:
            assert client.returncode == 0, client_log_path.read_text(
                encoding="utf-8", errors="replace"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=20)
        server_log.close()

    rollout_dir = ENV_CLIENT_ROOT / "runs" / exp_name / "rollout" / "proc_000"
    full_video_dir = rollout_dir / "videos" / "full" / "images"
    policy_first_obs_dir = policy_debug_dir / "first_observation"

    assert full_video_dir.is_dir(), f"Missing rollout video directory: {full_video_dir}"
    assert list(full_video_dir.glob("*.mp4")), f"No rollout mp4 files found in {full_video_dir}"
    assert policy_first_obs_dir.is_dir(), (
        f"Missing policy debug artifact directory: {policy_first_obs_dir}"
    )
    assert (policy_first_obs_dir / "tokenized_prompt.npy").is_file()
    assert (policy_first_obs_dir / "tokenized_prompt_decoded.txt").is_file()
    assert (policy_first_obs_dir / "summary.json").is_file()

    server_log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    client_log_text = client_log_path.read_text(encoding="utf-8", errors="replace")
    assert "Loading OpenPI model weights from" in server_log_text
    assert "OpenPI norm stats source:" in server_log_text
    assert "OpenPI input transform pipeline:" in server_log_text
    assert "Saved OpenPI debug artifacts to" in server_log_text
    if client_timed_out:
        assert "Intermediate rollout timing summary" in client_log_text, client_log_text
