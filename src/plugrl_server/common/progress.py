from __future__ import annotations

import dataclasses
import sys
import time
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from plugrl_server.common.metrics import MetricDict


@dataclasses.dataclass(frozen=True)
class ProgressSnapshot:
    global_step: int
    global_steps: int | None
    total_time: float
    collection_time: float
    learn_time: float
    fps: float
    eta_time: float | None

    def as_metrics(self) -> MetricDict:
        progress = dict(
            global_step=float(self.global_step),
            total_time=self.total_time,
            collection_time=self.collection_time,
            learn_time=self.learn_time,
            fps=self.fps,
        )
        if self.global_steps is not None:
            progress["global_steps"] = float(self.global_steps)
            progress["global_steps_remaining"] = float(
                max(0, self.global_steps - self.global_step)
            )
        if self.eta_time is not None:
            progress["eta_time"] = self.eta_time
        return dict(progress=progress)


class ProgressTracker:
    def __init__(self, *, global_steps: int | None = None) -> None:
        self._start_time = time.monotonic()
        self._collection_start_time = self._start_time
        self._global_steps = global_steps

    def mark_learn_start(self) -> float:
        return time.monotonic()

    def snapshot_after_learn(
        self, *, global_step: int, learn_started_at: float
    ) -> ProgressSnapshot:
        completed_at = time.monotonic()
        total_time = max(0.0, completed_at - self._start_time)
        collection_time = max(0.0, learn_started_at - self._collection_start_time)
        learn_time = max(0.0, completed_at - learn_started_at)
        fps = 0.0 if total_time <= 0 else float(global_step) / total_time

        eta_time = None
        if self._global_steps is not None and self._global_steps > 0:
            remaining_steps = max(0, self._global_steps - global_step)
            if remaining_steps == 0:
                eta_time = 0.0
            elif fps > 0:
                eta_time = float(remaining_steps) / fps

        self._collection_start_time = completed_at

        return ProgressSnapshot(
            global_step=global_step,
            global_steps=self._global_steps,
            total_time=total_time,
            collection_time=collection_time,
            learn_time=learn_time,
            fps=fps,
            eta_time=eta_time,
        )


class ProgressReporter:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled and sys.stderr.isatty()
        self._task_ids: dict[str, TaskID] = dict()
        if self._enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            )
            self._progress.start()
        else:
            self._progress = None

    def start_phase(
        self, phase: str, *, total: int | None, description: str | None = None
    ) -> None:
        if not self._enabled or self._progress is None:
            return
        self.finish_phase(phase)
        self._task_ids[phase] = self._progress.add_task(
            description or phase,
            total=total if total is not None else None,
        )

    def update_phase(
        self,
        phase: str,
        *,
        advance: int = 1,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if not self._enabled or self._progress is None:
            return
        task_id = self._task_ids.get(phase)
        if task_id is None:
            return
        kwargs = dict(advance=advance)
        if completed is not None:
            kwargs["completed"] = completed
        if total is not None:
            kwargs["total"] = total
        self._progress.update(task_id, **kwargs)

    def finish_phase(self, phase: str) -> None:
        if not self._enabled or self._progress is None:
            return
        task_id = self._task_ids.pop(phase, None)
        if task_id is None:
            return
        self._progress.remove_task(task_id)
        self._progress.refresh()

    def close(self) -> None:
        if not self._enabled or self._progress is None:
            return
        self._task_ids.clear()
        self._progress.stop()

    def get_console(self) -> Console | None:
        if not self._enabled or self._progress is None:
            return None
        return self._progress.console
