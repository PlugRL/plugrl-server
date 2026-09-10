"""Sweep behaviour that does not need a server: matrix construction, resume,
and the free-port helper.

The runner itself is exercised end to end by scripts/smoke.sh, which needs
real executables.
"""

import json

from plugrl_sweep.aggregate import load_results
from plugrl_sweep.cell import Cell
from plugrl_sweep.outcome import CellResult, Outcome
from plugrl_sweep.runner import free_port, wait_for_server
from plugrl_sweep.sweep import already_done, build_matrix, _coerce


class TestBuildMatrix:
    def test_cross_product_size(self):
        cells = build_matrix(["a", "b"], ["p", "q"], [0, 1, 2])
        assert len(cells) == 12

    def test_keys_are_unique(self):
        cells = build_matrix(["a", "b"], ["p", "q"], [0, 1, 2])
        assert len({c.key for c in cells}) == len(cells)

    def test_overrides_reach_every_cell(self):
        cells = build_matrix(["a"], ["p"], [0], policy_overrides={"action_dim": 7})
        assert cells[0].policy_overrides == {"action_dim": 7}

    def test_overrides_are_not_shared_between_cells(self):
        """A dict default would be aliased across the whole matrix."""
        cells = build_matrix(["a", "b"], ["p"], [0], policy_overrides={"x": 1})
        cells[0].policy_overrides["x"] = 99
        assert cells[1].policy_overrides["x"] == 1

    def test_per_env_overrides_layer_on_top(self):
        """CartPole needs a discrete policy; LIBERO needs a continuous one.
        One global override cannot serve both."""
        cells = build_matrix(
            ["classic-v1", "libero-v1"],
            ["dummy-policy"],
            [0],
            policy_overrides={"discrete": False, "action_dim": 7},
            per_env_policy_overrides={
                "classic-v1": {"discrete": True, "action_dim": 2}
            },
        )
        by_env = {c.env_uid: c.policy_overrides for c in cells}
        assert by_env["classic-v1"] == {"discrete": True, "action_dim": 2}
        assert by_env["libero-v1"] == {"discrete": False, "action_dim": 7}

    def test_per_env_overrides_do_not_leak(self):
        cells = build_matrix(
            ["a", "b"],
            ["p"],
            [0],
            per_env_policy_overrides={"a": {"x": 1}},
        )
        by_env = {c.env_uid: c.policy_overrides for c in cells}
        assert by_env["a"] == {"x": 1}
        assert by_env["b"] == {}


class TestResume:
    def test_no_file_means_nothing_done(self, tmp_path):
        assert already_done(tmp_path / "missing.jsonl") == set()

    def test_finished_cells_are_recognised(self, tmp_path):
        path = tmp_path / "results.jsonl"
        cell = Cell(env_uid="dummy-v1", policy_uid="dummy-policy", seed=0)
        result = CellResult(
            key=cell.key,
            env_uid=cell.env_uid,
            policy_uid=cell.policy_uid,
            algo_uid=cell.algo_uid,
            seed=cell.seed,
            outcome=Outcome.OK,
        )
        path.write_text(json.dumps(result.to_dict()) + "\n", encoding="utf-8")

        assert cell.key in already_done(path)
        assert load_results(path)[0].outcome is Outcome.OK


class TestFreePort:
    def test_returns_a_usable_port(self):
        port = free_port()
        assert 1024 < port < 65536

    def test_successive_calls_differ(self):
        """Not guaranteed by the OS, but a helper that always returned the
        same port would serialise the whole sweep onto one socket."""
        ports = {free_port() for _ in range(5)}
        assert len(ports) > 1


class TestWaitForServer:
    def test_detects_the_ready_marker(self, tmp_path):
        log = tmp_path / "server.log"
        log.write_text("Agent Server is listening on 0.0.0.0:8000\n", encoding="utf-8")

        class AliveProcess:
            def poll(self):
                return None

        assert wait_for_server(log, AliveProcess(), timeout=1.0) is True

    def test_gives_up_when_the_process_died(self, tmp_path):
        log = tmp_path / "server.log"
        log.write_text("Traceback...\n", encoding="utf-8")

        class DeadProcess:
            def poll(self):
                return 1

        assert wait_for_server(log, DeadProcess(), timeout=5.0) is False

    def test_times_out_on_a_silent_server(self, tmp_path):
        log = tmp_path / "server.log"
        log.write_text("starting up\n", encoding="utf-8")

        class AliveProcess:
            def poll(self):
                return None

        assert wait_for_server(log, AliveProcess(), timeout=0.3) is False


class TestCoerce:
    def test_bools(self):
        assert _coerce("true") is True
        assert _coerce("False") is False

    def test_numbers(self):
        assert _coerce("7") == 7
        assert _coerce("0.5") == 0.5

    def test_strings_pass_through(self):
        assert _coerce("hopper-medium-v2") == "hopper-medium-v2"
