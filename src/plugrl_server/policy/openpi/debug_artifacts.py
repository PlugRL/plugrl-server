from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Any

import numpy as np

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        summary = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            if np.issubdtype(value.dtype, np.number) or np.issubdtype(
                value.dtype, np.bool_
            ):
                summary["min"] = float(value.min())
                summary["max"] = float(value.max())
            else:
                summary["sample"] = value.reshape(-1)[0].item()
        else:
            summary["sample"] = None
        return summary
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strip_leading_unit_dims(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        current = value
        while current.ndim > 0 and current.shape[0] == 1:
            current = current[0]
        return current
    if isinstance(value, dict):
        return {key: _strip_leading_unit_dims(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_leading_unit_dims(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_leading_unit_dims(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@lru_cache(maxsize=1)
def _load_sentencepiece_tokenizer():
    import sentencepiece
    from openpi.shared import download as _download

    path = _download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    )
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.load(str(path))
    return tokenizer


def _decode_tokenized_prompt(token_ids: np.ndarray, mask: np.ndarray | None) -> str:
    active = np.asarray(token_ids)
    if mask is not None:
        active = active[np.asarray(mask, dtype=bool)]
    tokenizer = _load_sentencepiece_tokenizer()
    return tokenizer.decode(active.astype(np.int32).tolist())


def _save_image(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    except Exception as exc:
        logger.warning("Failed to save debug image %s: %s", path, exc)
        np.save(path.with_suffix(".npy"), np.asarray(image), allow_pickle=False)


def write_debug_artifacts(
    artifact_dir: pathlib.Path,
    *,
    raw_obs: dict[str, Any],
    transformed_obs: dict[str, Any],
) -> pathlib.Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_obs = _strip_leading_unit_dims(raw_obs)
    transformed_obs = _strip_leading_unit_dims(transformed_obs)

    summary = {
        "raw_obs": _to_builtin(raw_obs),
        "transformed_obs": _to_builtin(transformed_obs),
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if "prompt" in raw_obs:
        prompt_value = raw_obs["prompt"]
        if isinstance(prompt_value, bytes):
            prompt_value = prompt_value.decode("utf-8")
        (artifact_dir / "prompt.txt").write_text(str(prompt_value), encoding="utf-8")

    for key, value in raw_obs.get("images", {}).items():
        _save_image(artifact_dir / "raw_images" / f"{key}.png", np.asarray(value))

    if "state" in raw_obs:
        with (artifact_dir / "raw_state.npy").open("wb") as file_obj:
            np.save(file_obj, np.asarray(raw_obs["state"]), allow_pickle=False)

    for key, value in transformed_obs.get("image", {}).items():
        _save_image(
            artifact_dir / "transformed_images" / f"{key}.png", np.asarray(value)
        )

    if "state" in transformed_obs:
        with (artifact_dir / "transformed_state.npy").open("wb") as file_obj:
            np.save(
                file_obj, np.asarray(transformed_obs["state"]), allow_pickle=False
            )

    tokenized_prompt = transformed_obs.get("tokenized_prompt")
    tokenized_prompt_mask = transformed_obs.get("tokenized_prompt_mask")
    if tokenized_prompt is not None:
        tokenized_prompt_arr = np.asarray(tokenized_prompt)
        with (artifact_dir / "tokenized_prompt.npy").open("wb") as file_obj:
            np.save(file_obj, tokenized_prompt_arr, allow_pickle=False)
        if tokenized_prompt_mask is not None:
            with (artifact_dir / "tokenized_prompt_mask.npy").open("wb") as file_obj:
                np.save(
                    file_obj,
                    np.asarray(tokenized_prompt_mask, dtype=np.bool_),
                    allow_pickle=False,
                )
        try:
            decoded = _decode_tokenized_prompt(tokenized_prompt_arr, tokenized_prompt_mask)
        except Exception as exc:
            decoded = f"<decode unavailable: {exc}>"
            logger.warning("Failed to decode tokenized prompt: %s", exc)
        (artifact_dir / "tokenized_prompt_decoded.txt").write_text(
            decoded, encoding="utf-8"
        )

    return artifact_dir
