from __future__ import annotations

import dataclasses

from plugrl_server.common.metrics import MetricDict


@dataclasses.dataclass
class RuntimeMetricTracker:
    ema_decay: float = 0.9
    infer_batch_time: float = 0.0
    infer_batch_size: float = 0.0
    feedback_batch_time: float = 0.0
    feedback_batch_size: float = 0.0

    def observe_infer(self, *, duration: float, batch_size: int) -> None:
        self.infer_batch_time = _ema(
            self.infer_batch_time, float(duration), decay=self.ema_decay
        )
        self.infer_batch_size = _ema(
            self.infer_batch_size, float(batch_size), decay=self.ema_decay
        )

    def observe_feedback(self, *, duration: float, batch_size: int) -> None:
        self.feedback_batch_time = _ema(
            self.feedback_batch_time, float(duration), decay=self.ema_decay
        )
        self.feedback_batch_size = _ema(
            self.feedback_batch_size, float(batch_size), decay=self.ema_decay
        )

    def as_metrics(self) -> MetricDict:
        return dict(
            runtime=dict(
                infer_batch_time=self.infer_batch_time,
                infer_batch_size=self.infer_batch_size,
                feedback_batch_time=self.feedback_batch_time,
                feedback_batch_size=self.feedback_batch_size,
            )
        )


def _ema(previous: float, current: float, *, decay: float) -> float:
    if previous <= 0:
        return current
    return decay * previous + (1.0 - decay) * current
