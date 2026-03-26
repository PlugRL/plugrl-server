from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)


class MetricSink(Protocol):
    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NoOpMetricSink:
    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class TensorBoardMetricSink:
    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def init_metric_sink_by_tracker(
    args: Any, *, resuming: bool, log_code: bool, enabled: bool = True
) -> tuple[MetricSink, Any]:
    if not enabled:
        tracker = None
    else:
        if args.track.tracker == "wandb":
            import wandb

            tracker_module = wandb
        else:
            import swanlab

            tracker_module = swanlab
            swanlab.sync_tensorboard_torch()

        ckpt_dir = args.checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        project_name = f"{args.track.project_name}-{args.algo_uid}-{args.policy_uid}"
        if resuming:
            run_id = (ckpt_dir / "run_id.txt").read_text().strip()
            tracker = tracker_module.init(
                id=run_id,
                resume="must",
                project=project_name,
                entity=args.track.entity or None,
                sync_tensorboard=True,
            )
        else:
            tracker = tracker_module.init(
                entity=args.track.entity or None,
                project=project_name,
                name=args.full_exp_name,
                config=dataclasses.asdict(args),
                save_code=log_code,
                sync_tensorboard=True,
            )
            run_id_value = tracker.id
            if run_id_value is None:
                raise RuntimeError("Tracker did not return a run id")
            (ckpt_dir / "run_id.txt").write_text(run_id_value)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(args.checkpoint_dir / "tensorboard"))
    return TensorBoardMetricSink(writer), tracker


def close_metric_sink_and_tracker(metric_sink: MetricSink, tracker: Any) -> None:
    try:
        metric_sink.flush()
    except Exception as exc:
        logger.warning(f"Failed to flush metric sink during shutdown: {exc}")

    try:
        metric_sink.close()
    except Exception as exc:
        logger.warning(f"Failed to close metric sink during shutdown: {exc}")

    if tracker is not None:
        try:
            tracker.finish()
        except Exception as exc:
            logger.warning(f"Failed to finish tracker during shutdown: {exc}")
