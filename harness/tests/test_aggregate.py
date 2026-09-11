import json

from plugrl_sweep.aggregate import (
    combine_seeds,
    compatibility_matrix,
    load_results,
    render_matrix,
    render_timing,
    timing_table,
)
from plugrl_sweep.outcome import CellResult, Outcome


def result(env, policy, outcome, seed=0, timing=None):
    return CellResult(
        key=f"{env}__{policy}__dummy__s{seed}",
        env_uid=env,
        policy_uid=policy,
        algo_uid="dummy",
        seed=seed,
        outcome=outcome,
        episodes_completed=2 if outcome is Outcome.OK else 0,
        timing=timing,
    )


class TestCombineSeeds:
    def test_all_ok_is_ok(self):
        assert combine_seeds([Outcome.OK] * 3) is Outcome.OK

    def test_one_failure_sinks_the_combination(self):
        """Two of three seeds working is not a working combination."""
        assert combine_seeds([Outcome.OK, Outcome.OK, Outcome.FAILED]) is Outcome.FAILED

    def test_unavailable_survives_only_if_unanimous(self):
        assert combine_seeds([Outcome.UNAVAILABLE] * 3) is Outcome.UNAVAILABLE

    def test_partly_unavailable_reports_what_was_observed(self):
        """If some seeds ran, "not installed" is not the story."""
        assert (
            combine_seeds([Outcome.UNAVAILABLE, Outcome.OK, Outcome.OK]) is Outcome.OK
        )
        assert combine_seeds([Outcome.UNAVAILABLE, Outcome.FAILED]) is Outcome.FAILED

    def test_failure_outranks_timeout_and_partial(self):
        assert (
            combine_seeds([Outcome.TIMEOUT, Outcome.FAILED, Outcome.INCOMPLETE])
            is Outcome.FAILED
        )

    def test_no_seeds_at_all(self):
        assert combine_seeds([]) is Outcome.UNAVAILABLE


class TestMatrix:
    def test_axes_and_cells(self):
        results = [
            result("atari-v1", "dummy-policy", Outcome.FAILED),
            result("dummy-v1", "dummy-policy", Outcome.OK),
            result("dummy-v1", "dppo-policy", Outcome.UNAVAILABLE),
        ]
        envs, policies, cells = compatibility_matrix(results)
        assert envs == ["atari-v1", "dummy-v1"]
        assert policies == ["dppo-policy", "dummy-policy"]
        assert cells[("dummy-v1", "dummy-policy")] is Outcome.OK
        assert cells[("atari-v1", "dummy-policy")] is Outcome.FAILED

    def test_seeds_fold_into_one_cell(self):
        results = [
            result("dummy-v1", "dummy-policy", Outcome.OK, seed=s) for s in range(3)
        ]
        _, _, cells = compatibility_matrix(results)
        assert len(cells) == 1

    def test_render_marks_unavailable_differently_from_failed(self):
        text = render_matrix(
            [
                result("a-v1", "p", Outcome.UNAVAILABLE),
                result("b-v1", "p", Outcome.FAILED),
            ]
        )
        assert "-" in text and "fail" in text

    def test_render_handles_no_results(self):
        assert render_matrix([]) == "(no results)"


class TestTiming:
    def test_only_ok_runs_contribute(self):
        rows = timing_table(
            [
                result("d", "p", Outcome.OK, seed=0, timing={"effective_fps": 10.0}),
                result(
                    "d", "p", Outcome.FAILED, seed=1, timing={"effective_fps": 999.0}
                ),
            ]
        )
        assert len(rows) == 1
        assert rows[0]["seeds"] == 1
        assert rows[0]["effective_fps"] == 10.0

    def test_median_across_seeds(self):
        rows = timing_table(
            [
                result("d", "p", Outcome.OK, seed=s, timing={"effective_fps": fps})
                for s, fps in enumerate([10.0, 20.0, 90.0])
            ]
        )
        assert rows[0]["effective_fps"] == 20.0

    def test_missing_fields_stay_none(self):
        rows = timing_table(
            [result("d", "p", Outcome.OK, timing={"effective_fps": 1.0})]
        )
        assert rows[0]["env_step_frac"] is None

    def test_render_handles_nothing_completed(self):
        assert render_timing([result("d", "p", Outcome.FAILED)]) == (
            "(no completed runs to time)"
        )


def test_load_results_round_trip(tmp_path):
    path = tmp_path / "results.jsonl"
    results = [
        result("dummy-v1", "dummy-policy", Outcome.OK, seed=0),
        result("atari-v1", "dummy-policy", Outcome.UNAVAILABLE, seed=0),
    ]
    path.write_text(
        "\n".join(json.dumps(r.to_dict()) for r in results), encoding="utf-8"
    )
    assert load_results(path) == results


def test_load_results_ignores_blank_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    r = result("dummy-v1", "dummy-policy", Outcome.OK)
    path.write_text(json.dumps(r.to_dict()) + "\n\n", encoding="utf-8")
    assert len(load_results(path)) == 1
