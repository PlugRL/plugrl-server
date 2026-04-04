from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from openpi.shared import download as _download
from openpi.shared import normalize as _normalize

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)


def _format_optional_array(value: Any) -> str:
    if value is None:
        return "None"
    array = np.asarray(value)
    return np.array2string(array, precision=6, separator=", ")


def _log_norm_stats(norm_stats: dict[str, _normalize.NormStats]) -> None:
    if not norm_stats:
        logger.info("Loaded RoboCasa norm stats are empty.")
        return

    logger.info("Loaded RoboCasa norm stats keys: %s", sorted(norm_stats))
    for key, stats in norm_stats.items():
        mean = np.asarray(stats.mean)
        std = np.asarray(stats.std)
        logger.info(
            "norm_stats[%s]: mean_shape=%s std_shape=%s q01_present=%s q99_present=%s",
            key,
            list(mean.shape),
            list(std.shape),
            stats.q01 is not None,
            stats.q99 is not None,
        )
        logger.info("norm_stats[%s].mean=%s", key, _format_optional_array(stats.mean))
        logger.info("norm_stats[%s].std=%s", key, _format_optional_array(stats.std))
        logger.info("norm_stats[%s].q01=%s", key, _format_optional_array(stats.q01))
        logger.info("norm_stats[%s].q99=%s", key, _format_optional_array(stats.q99))


def _normalize_robocasa_policy_norm_stats(
    train_config: Any,
    norm_stats: dict[str, _normalize.NormStats],
) -> dict[str, _normalize.NormStats]:
    transformed = train_config.data._transform_loaded_norm_stats(norm_stats)
    return {
        key: value for key, value in transformed.items() if key in {"state", "actions"}
    }


def _load_norm_stats_path(path: pathlib.Path) -> dict[str, _normalize.NormStats]:
    downloaded = _download.maybe_download(str(path))
    return _normalize.load(downloaded)


def _maybe_shared_norm_stats_path(train_config: Any) -> str | None:
    data_factory = train_config.data
    use_shared = getattr(data_factory, "use_shared_norm_stats", False)
    if not use_shared:
        return None
    shared_path = getattr(data_factory, "shared_norm_stats_path", None)
    if not shared_path:
        logger.warning(
            "Config %s enables shared norm stats but missing shared_norm_stats_path.",
            train_config.name,
        )
        return None
    return shared_path


def load_robocasa_norm_stats(
    train_config: Any,
    *,
    dataset_dir: pathlib.Path | None,
    norm_stats_path: pathlib.Path | None,
) -> tuple[dict[str, _normalize.NormStats], pathlib.Path]:
    if norm_stats_path is not None:
        raw_norm_stats = _load_norm_stats_path(norm_stats_path)
        normalized = _normalize_robocasa_policy_norm_stats(train_config, raw_norm_stats)
        logger.info("Loaded RoboCasa norm stats from explicit path: %s", norm_stats_path)
        _log_norm_stats(normalized)
        return normalized, norm_stats_path

    shared_path = _maybe_shared_norm_stats_path(train_config)
    if shared_path is not None:
        shared_path_obj = pathlib.Path(shared_path)
        try:
            raw_norm_stats = _load_norm_stats_path(shared_path_obj)
        except FileNotFoundError:
            logger.warning(
                "Shared norm stats path %s could not be loaded. Falling back to dataset-specific stats.",
                shared_path,
            )
        else:
            normalized = _normalize_robocasa_policy_norm_stats(
                train_config, raw_norm_stats
            )
            logger.info("Loaded shared RoboCasa norm stats from %s", shared_path)
            _log_norm_stats(normalized)
            return normalized, shared_path_obj

    if dataset_dir is None:
        raise ValueError(
            "RoboCasa policy requires either dataset_dir, norm_stats_path, or a config with shared norm stats."
        )

    dataset_path = pathlib.Path(dataset_dir)
    candidate_dirs = [dataset_path / "meta", dataset_path]
    for candidate in candidate_dirs:
        try:
            raw_norm_stats = _load_norm_stats_path(candidate)
        except FileNotFoundError:
            logger.info(
                "RoboCasa norm stats not found in %s, trying next candidate.",
                candidate,
            )
            continue

        normalized = _normalize_robocasa_policy_norm_stats(train_config, raw_norm_stats)
        logger.info("Loaded RoboCasa norm stats from %s", candidate)
        _log_norm_stats(normalized)
        return normalized, candidate

    raise FileNotFoundError(
        "Could not load RoboCasa norm stats from dataset_dir. Expected stats in either "
        f"{dataset_path / 'meta'} or {dataset_path}."
    )
