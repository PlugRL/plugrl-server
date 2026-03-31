from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Protocol, TypeAlias
from rich.console import Console
from rich.table import Table
from rich import box

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)
_TABLE_CONSOLE = Console(highlight=False)

MetricScalar: TypeAlias = float | int
MetricValue: TypeAlias = MetricScalar | Mapping[str, "MetricValue"]
MetricDict: TypeAlias = Mapping[str, MetricValue]
MutableMetricDict: TypeAlias = dict[str, MetricValue]


def merge_metric_groups(*metric_groups: Mapping[str, MetricValue]) -> MetricDict:
    merged: MutableMetricDict = dict()
    for metric_group in metric_groups:
        _merge_metric_group_into(merged, metric_group)
    return merged


def _merge_metric_group_into(
    target: MutableMetricDict, source: Mapping[str, MetricValue]
) -> None:
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_metric_group_into(existing, value)
        elif isinstance(value, Mapping):
            nested: MutableMetricDict = dict()
            _merge_metric_group_into(nested, value)
            target[key] = nested
        else:
            target[key] = value


def flatten_metrics(
    metrics: Mapping[str, MetricValue], *, prefix: str = ""
) -> dict[str, float]:
    flattened = dict()
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=full_key))
        else:
            flattened[full_key] = float(value)
    return flattened


class MetricSink(Protocol):
    def log_scalars(self, scalars: MetricDict, *, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NoOpMetricSink:
    def log_scalars(self, scalars: MetricDict, *, step: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class TensorBoardMetricSink:
    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def log_scalars(self, scalars: MetricDict, *, step: int) -> None:
        for key, value in flatten_metrics(scalars).items():
            self._writer.add_scalar(key, value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def build_metrics_table(metrics: Mapping[str, MetricValue]) -> Table | None:
    rows = _build_single_column_metric_rows(metrics)
    if not rows:
        return None

    title = _build_metrics_table_title(metrics)
    table = Table(
        box=box.ROUNDED,
        show_header=False,
        expand=False,
        pad_edge=False,
        collapse_padding=True,
        title=title,
        title_justify="left",
    )
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    for key, value in rows:
        table.add_row(_style_metric_key(key), value)
    return table


def format_metrics_table(metrics: Mapping[str, MetricValue]) -> str:
    table = build_metrics_table(metrics)
    if table is None:
        return ""
    console = Console(
        width=120,
        record=True,
        force_terminal=False,
        color_system=None,
    )
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip()


def render_metrics_table(
    metrics: Mapping[str, MetricValue], *, console: Console | None = None
) -> bool:
    table = build_metrics_table(metrics)
    if table is None:
        return False
    (console or _TABLE_CONSOLE).print(table)
    return True


def _build_single_column_metric_rows(
    metrics: Mapping[str, MetricValue],
) -> list[tuple[str, str]]:
    group_names = (
        "rollout",
        "train",
        "losses",
        "models",
        "server",
        "runtime",
        "learn_runtime",
        "progress",
    )
    return _flatten_metric_groups(metrics, group_names)


def _build_metrics_table_title(metrics: Mapping[str, MetricValue]) -> str | None:
    progress = metrics.get("progress")
    if not isinstance(progress, Mapping):
        return None

    global_step = progress.get("global_step")
    global_steps = progress.get("global_steps")
    if global_step is None:
        return None
    if global_steps is None:
        return f"Step {int(float(global_step))}"
    return f"Step {int(float(global_step))} / {int(float(global_steps))}"


def _flatten_metric_groups(
    metrics: Mapping[str, MetricValue], group_names: tuple[str, ...]
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for group_name in group_names:
        group_value = metrics.get(group_name)
        if isinstance(group_value, Mapping):
            rows.extend(_flatten_metric_group(group_name, group_value))
    return rows


def _flatten_metric_group(
    group_name: str, group_value: Mapping[str, MetricValue]
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.append((f"{group_name}/", ""))
    for metric_name, metric_value in group_value.items():
        if isinstance(metric_value, Mapping):
            rows.append((f" {metric_name}/", ""))
            for nested_name, nested_value in metric_value.items():
                rows.append(
                    (
                        f"  {nested_name}",
                        _format_metric_value(nested_name, nested_value),
                    )
                )
        else:
            rows.append(
                (
                    f" {metric_name}",
                    _format_metric_value(metric_name, metric_value),
                )
            )
    return rows


def _style_metric_key(metric_key: str) -> str:
    if not metric_key:
        return ""
    if metric_key.endswith("/"):
        return f"[bold cyan]{metric_key}[/bold cyan]"
    return metric_key


def _format_metric_value(metric_name: str, value: MetricValue) -> str:
    if isinstance(value, Mapping):
        return ""
    formatted_value: str
    float_value = float(value)
    if metric_name.endswith("time"):
        formatted_value = _format_duration(float_value)
    elif metric_name.endswith("step") or metric_name.endswith("steps"):
        formatted_value = str(int(float_value))
    elif metric_name.endswith("remaining"):
        formatted_value = str(int(float_value))
    elif abs(float_value) >= 1000 and float_value.is_integer():
        formatted_value = str(int(float_value))
    else:
        formatted_value = f"{float_value:.6g}"
    return formatted_value


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
