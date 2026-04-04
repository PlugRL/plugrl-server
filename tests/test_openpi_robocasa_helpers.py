from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from openpi.shared.normalize import NormStats

from plugrl_server.policy.openpi.debug_artifacts import write_debug_artifacts
from plugrl_server.policy.openpi.openpi_transforming import get_transform
from plugrl_server.policy.openpi.robocasa_norm_stats import load_robocasa_norm_stats


def test_get_transform_maps_plugrl_robocasa_observation():
    transform = get_transform("pi05_robocasa_test")[0]
    obs = {
        "images": {
            "robot0_agentview_left": np.full((8, 8, 3), 1, dtype=np.uint8),
            "robot0_agentview_right": np.full((8, 8, 3), 2, dtype=np.uint8),
            "robot0_eye_in_hand": np.full((8, 8, 3), 3, dtype=np.uint8),
        },
        "states": {"state": np.arange(16, dtype=np.float32)},
        "text": "close the blender lid",
    }

    repacked = transform(obs)

    assert repacked["prompt"] == "close the blender lid"
    np.testing.assert_array_equal(repacked["state"], np.arange(16, dtype=np.float32))
    np.testing.assert_array_equal(
        repacked["images"]["robot0_eye_in_hand"],
        np.full((8, 8, 3), 3, dtype=np.uint8),
    )


def test_write_debug_artifacts_exports_prompt_tokens_and_images(
    tmp_path: pathlib.Path, monkeypatch
):
    class _FakeTokenizer:
        def decode(self, token_ids):
            return "decoded:" + ",".join(map(str, token_ids))

    monkeypatch.setattr(
        "plugrl_server.policy.openpi.debug_artifacts._load_sentencepiece_tokenizer",
        lambda: _FakeTokenizer(),
    )

    raw_obs = {
        "images": {
            "robot0_agentview_left": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "robot0_agentview_right": np.ones((1, 4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.full((1, 4, 4, 3), 2, dtype=np.uint8),
        },
        "state": np.arange(16, dtype=np.float32)[None, :],
        "prompt": np.asarray(["close the blender lid"]),
    }
    transformed_obs = {
        "image": {
            "base_0_rgb": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.ones((1, 4, 4, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.full((1, 4, 4, 3), 2, dtype=np.uint8),
        },
        "state": np.arange(16, dtype=np.float32)[None, :],
        "tokenized_prompt": np.asarray([[1, 2, 3, 0]], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray([[True, True, True, False]]),
    }

    artifact_dir = write_debug_artifacts(
        tmp_path / "policy_debug",
        raw_obs=raw_obs,
        transformed_obs=transformed_obs,
    )

    assert (artifact_dir / "summary.json").is_file()
    assert (artifact_dir / "prompt.txt").read_text(encoding="utf-8") == (
        "close the blender lid"
    )
    assert (artifact_dir / "tokenized_prompt.npy").is_file()
    assert (artifact_dir / "tokenized_prompt_decoded.txt").read_text(
        encoding="utf-8"
    ) == "decoded:1,2,3"
    assert (artifact_dir / "raw_images" / "robot0_eye_in_hand.png").is_file()
    assert (artifact_dir / "transformed_images" / "base_0_rgb.png").is_file()


def test_load_robocasa_norm_stats_prefers_explicit_path(caplog, monkeypatch):
    explicit_path = pathlib.Path("/tmp/robocasa_norm_stats.json")
    norm_stats = {
        "observation.state": NormStats(
            mean=np.zeros(16, dtype=np.float32),
            std=np.ones(16, dtype=np.float32),
            q01=np.full(16, -1.0, dtype=np.float32),
            q99=np.full(16, 1.0, dtype=np.float32),
        ),
        "action": NormStats(
            mean=np.zeros(12, dtype=np.float32),
            std=np.ones(12, dtype=np.float32),
            q01=np.full(12, -1.0, dtype=np.float32),
            q99=np.full(12, 1.0, dtype=np.float32),
        ),
    }

    @dataclasses.dataclass
    class _FakeDataFactory:
        use_shared_norm_stats: bool = True
        shared_norm_stats_path: str | None = "/tmp/shared_norm_stats.json"

        def _transform_loaded_norm_stats(self, loaded):
            return {
                "state": loaded["observation.state"],
                "actions": loaded["action"],
            }

    @dataclasses.dataclass
    class _FakeTrainConfig:
        name: str = "pi05_robocasa_test"
        data: _FakeDataFactory = dataclasses.field(default_factory=_FakeDataFactory)

    monkeypatch.setattr(
        "plugrl_server.policy.openpi.robocasa_norm_stats._load_norm_stats_path",
        lambda path: norm_stats,
    )

    caplog.set_level("INFO")
    loaded, source = load_robocasa_norm_stats(
        _FakeTrainConfig(),
        dataset_dir=pathlib.Path("/unused/dataset"),
        norm_stats_path=explicit_path,
    )

    assert source == explicit_path
    assert sorted(loaded.keys()) == ["actions", "state"]
    assert "Loaded RoboCasa norm stats from explicit path" in caplog.text
