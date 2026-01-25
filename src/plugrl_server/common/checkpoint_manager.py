import gc
import shutil
import time
import dataclasses
import yaml
from pathlib import Path
from typing import Optional, Any

from loguru import logger

import safetensors.torch
import torch


def get_latest_checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    if not checkpoint_dir.exists():
        return None
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def move_model_to_cpu(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.cpu() for k, v in model_state.items()}


def move_optimizer_to_cpu(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    def move_to_cpu(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cpu(v) for v in obj]
        else:
            return obj

    return move_to_cpu(optimizer_state)


@dataclasses.dataclass
class Checkpoint:
    step: int
    model: dict[str, torch.Tensor] | None = None
    optimizer: dict[str, Any] | None = None
    meta: dict = dataclasses.field(default_factory=lambda: {})


def load_checkpoint_from_path(checkpoint_path: Path, step=0) -> Checkpoint:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    checkpoint = Checkpoint(step=step)

    try:
        logger.info(f"Loading model state from {checkpoint_path}...")
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            checkpoint.model = safetensors.torch.load_file(safetensors_path)

        logger.info(f"Loading optimizer state from {checkpoint_path}...")
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint.optimizer = torch.load(
                optimizer_path, map_location="cpu", weights_only=False
            )

        logger.info(f"Loading metadata from {checkpoint_path}...")
        metadata_path = checkpoint_path / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
            checkpoint.meta = metadata.get("checkpoint_meta", {})
            checkpoint.step = metadata.get("step", 0)
        else:
            logger.warning("Metadata file not found. Resuming step may be inaccurate.")

        logger.info(f"Checkpoint loaded from {checkpoint_path}.")
        return checkpoint
    except Exception as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            logger.error(
                "Out of memory error while loading checkpoint. Try reducing batch size or using a machine with more memory."
            )
        raise


class CheckpointManager:
    checkpoint_dir: Path
    config: dict

    def __init__(
        self, checkpoint_dir: Path, *, config: dict, overwrite: bool, resume: bool
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        if self.checkpoint_dir.exists():
            if overwrite:
                logger.warning(
                    f"Overwriting existing checkpoint directory: {self.checkpoint_dir}"
                )
                shutil.rmtree(self.checkpoint_dir)
            elif resume:
                latest_step = get_latest_checkpoint_step(self.checkpoint_dir)
                if latest_step is not None:
                    logger.info(
                        f"Resuming from latest checkpoint at step {latest_step} in {self.checkpoint_dir}"
                    )
                else:
                    logger.info(
                        f"No checkpoints found in {self.checkpoint_dir}. Starting fresh."
                    )
            else:
                raise FileExistsError(
                    f"Checkpoint directory {self.checkpoint_dir} already exists. Use overwrite or resume options."
                )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: Checkpoint):
        step = checkpoint.step

        final_ckpt_dir = self.checkpoint_dir / str(step)
        tmp_ckpt_dir = self.checkpoint_dir / f"tmp_{step}"

        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=False)

        if checkpoint.model:
            model_state_dict = checkpoint.model
            safe_state_dict = {
                k: v.cpu().clone().detach() for k, v in model_state_dict.items()
            }
            safetensors.torch.save_file(
                safe_state_dict, tmp_ckpt_dir / "model.safetensors"
            )
            del checkpoint.model

        if checkpoint.optimizer:
            torch.save(checkpoint.optimizer, tmp_ckpt_dir / "optimizer.pt")
            del checkpoint.optimizer

        metadata = dict(
            step=step,
            config=self.config,
            timestamp=time.time(),
            checkpoint_meta=checkpoint.meta,
        )
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        yaml_path = tmp_ckpt_dir / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(self.config, f, indent=4, sort_keys=False)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logger.info(f"Checkpoint saved at step {step} to {final_ckpt_dir}")

    def load_checkpoint(self, step: Optional[int] = None) -> Optional[Checkpoint]:
        if step is None:
            step = get_latest_checkpoint_step(self.checkpoint_dir)
            if step is None:
                logger.info("No checkpoints found.")
                return None

        ckpt_dir = self.checkpoint_dir / str(step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        checkpoint = load_checkpoint_from_path(ckpt_dir, step=step)
        return checkpoint
