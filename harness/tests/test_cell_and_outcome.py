import pytest

from plugrl_sweep.cell import Cell
from plugrl_sweep.outcome import CellResult, Outcome, classify, final_exception


class TestCell:
    def test_key_is_stable_and_includes_the_seed(self):
        a = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=3)
        b = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=3)
        c = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=4)
        assert a.key == b.key
        assert a.key != c.key

    def test_combo_ignores_the_seed(self):
        a = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=3)
        b = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=9)
        assert a.combo == b.combo == ("dummy-v1", "dummy-policy")

    def test_server_command_shape(self):
        cell = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=7)
        cmd = cell.server_command("plugrl-run-server", port=9001)
        assert cmd[:5] == [
            "plugrl-run-server",
            "dummy-policy",
            "default",
            "dummy",
            "default",
        ]
        assert "--port" in cmd and "9001" in cmd
        assert "--seed" in cmd and "7" in cmd

    def test_client_command_carries_the_seed(self):
        cell = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=11)
        cmd = cell.client_command("plugrl-run-env-client", port=9001, host="127.0.0.1")
        assert cmd[1] == "dummy-v1"
        assert "--runner.seed" in cmd
        assert cmd[cmd.index("--runner.seed") + 1] == "11"

    def test_numeric_overrides_render_as_value_flags(self):
        cell = Cell(
            env_uid="dummy-v1",
            policy_uid="dummy-policy",
            policy_overrides={"action_dim": 7},
        )
        cmd = cell.server_command("plugrl-run-server", port=1)
        assert "--policy.action-dim" in cmd
        assert cmd[cmd.index("--policy.action-dim") + 1] == "7"

    @pytest.mark.parametrize(
        "value,expected", [(False, "--policy.no-discrete"), (True, "--policy.discrete")]
    )
    def test_bool_overrides_render_as_tyro_flag_pairs(self, value, expected):
        """tyro takes --x/--no-x for bools; a bare "False" would be positional."""
        cell = Cell(
            env_uid="dummy-v1",
            policy_uid="dummy-policy",
            policy_overrides={"discrete": value},
        )
        cmd = cell.server_command("plugrl-run-server", port=1)
        assert expected in cmd
        assert "False" not in cmd and "True" not in cmd


class TestClassify:
    def test_clean_run(self):
        assert (
            classify(
                returncode=0,
                stdout="",
                episodes_expected=2,
                episodes_completed=2,
            )
            is Outcome.OK
        )

    def test_unregistered_env_is_unavailable(self):
        """tyro rejects the uid outright when the extra never registered it."""
        outcome = classify(
            returncode=1,
            stdout=(
                "Argument {dummy-v1,atari-v1}: invalid choice: 'libero-v1' "
                "(choose from 'dummy-v1', 'atari-v1')"
            ),
            episodes_expected=2,
            episodes_completed=None,
            env_uid="libero-v1",
        )
        assert outcome is Outcome.UNAVAILABLE
        assert not outcome.is_evidence

    def test_startup_error_is_a_failure(self):
        outcome = classify(
            returncode=1,
            stdout="ValueError: Env must expose action space",
            episodes_expected=2,
            episodes_completed=None,
            env_uid="classic-v1",
        )
        assert outcome is Outcome.FAILED
        assert outcome.is_evidence

    def test_startup_warnings_about_other_envs_do_not_mask_a_failure(self):
        """The bug this classifier shipped with.

        The env client warns about every extra it could not import, on every
        run, so a log always contains "d4rl is not installed" no matter which
        env was asked for. Matching on that marked a genuinely failing cell
        unavailable, and the whole matrix with it.
        """
        stdout = "\n".join(
            [
                "WARNING Skip loading env module ...d4rl_env: d4rl is not installed.",
                "WARNING Skip loading env module ...libero_env: libero is not installed.",
                "Traceback (most recent call last):",
                "ValueError: Expected action shape tail (), got (7,)",
            ]
        )
        outcome = classify(
            returncode=1,
            stdout=stdout,
            episodes_expected=2,
            episodes_completed=None,
            env_uid="classic-v1",
        )
        assert outcome is Outcome.FAILED, (
            "warnings about other envs are not evidence about this one"
        )

    def test_missing_runtime_dependency_is_unavailable(self):
        """atari-v1 registers, then cannot build its env without opencv.

        The name was accepted, so the "invalid choice" signal does not fire,
        but the machine is still missing something. That is not evidence
        about whether the combination works.
        """
        stdout = "\n".join(
            [
                "WARNING Skip loading env module ...d4rl_env: d4rl is not installed.",
                "  File gymnasium/wrappers/transform_observation.py, line 384",
                "    raise DependencyNotInstalled(",
                "gymnasium.error.DependencyNotInstalled: opencv (cv2) is "
                'not installed, run `pip install "gymnasium[other]"`',
            ]
        )
        outcome = classify(
            returncode=1,
            stdout=stdout,
            episodes_expected=2,
            episodes_completed=None,
            env_uid="atari-v1",
        )
        assert outcome is Outcome.UNAVAILABLE

    def test_import_error_naming_an_extra_is_unavailable(self):
        stdout = "ImportError: libero is not installed. Install the libero extra."
        assert (
            classify(
                returncode=1,
                stdout=stdout,
                episodes_expected=2,
                episodes_completed=None,
                env_uid="libero-v1",
            )
            is Outcome.UNAVAILABLE
        )

    def test_other_import_errors_are_failures(self):
        """An ImportError that is not about a missing extra is a real bug."""
        stdout = "ImportError: cannot import name 'Foo' from 'plugrl_env_client'"
        assert (
            classify(
                returncode=1,
                stdout=stdout,
                episodes_expected=2,
                episodes_completed=None,
                env_uid="dummy-v1",
            )
            is Outcome.FAILED
        )

    def test_unavailable_needs_this_envs_name(self):
        """A rejection naming a different env says nothing about this one."""
        outcome = classify(
            returncode=1,
            stdout="invalid choice: 'something-else'",
            episodes_expected=2,
            episodes_completed=None,
            env_uid="classic-v1",
        )
        assert outcome is Outcome.FAILED

    def test_exit_zero_without_episodes_is_incomplete(self):
        """The mistake that reported 118879 exchanges/s from a dead client."""
        assert (
            classify(
                returncode=0,
                stdout="",
                episodes_expected=2,
                episodes_completed=0,
            )
            is Outcome.INCOMPLETE
        )

    def test_partial_episodes_are_incomplete(self):
        assert (
            classify(returncode=0, stdout="", episodes_expected=5, episodes_completed=3)
            is Outcome.INCOMPLETE
        )

    def test_timeout_beats_everything(self):
        assert (
            classify(
                returncode=0,
                stdout="",
                episodes_expected=2,
                episodes_completed=2,
                timed_out=True,
            )
            is Outcome.TIMEOUT
        )

    def test_only_ok_counts_as_runnable(self):
        runnable = [o for o in Outcome if o.counts_as_runnable]
        assert runnable == [Outcome.OK]


class TestCellResult:
    def test_round_trips_through_dict(self):
        result = CellResult(
            key="k",
            env_uid="dummy-v1",
            policy_uid="dummy-policy",
            algo_uid="dummy",
            seed=1,
            outcome=Outcome.OK,
            duration_s=1.5,
            episodes_completed=2,
            timing={"effective_fps": 10.0},
        )
        restored = CellResult.from_dict(result.to_dict())
        assert restored == result
        assert restored.outcome is Outcome.OK


class TestFinalException:
    def test_reads_the_last_exception_not_the_first_mention(self):
        stdout = "\n".join(
            [
                "WARNING d4rl is not installed.",
                "ModuleNotFoundError: No module named 'something_early'",
                "During handling of the above exception, another occurred:",
                "ValueError: Expected action shape tail (), got (7,)",
            ]
        )
        assert final_exception(stdout) == (
            "ValueError",
            "Expected action shape tail (), got (7,)",
        )

    def test_none_when_there_is_no_traceback(self):
        assert final_exception("everything went fine\n") is None
