import dataclasses
import numpy as np
import time
import torch
from loguru import logger

from .base_algorithm import DDPAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from plugrl_server.policy.base_policy import InternalState, BasePolicy
from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.data_utils import unbatch_aggregate

UID = "dummy"


@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    fake_inference_duration_sec: float = 0.1
    fake_learn_duration_sec: float = 10.0
    fake_learn_freq: int = 100
    fake_total_steps: int = 300
    verbose: bool = True
    verbose_feedback_interval: int = 10

    break_action_chunk: bool = False


@register_algo(UID)
class DummyAlgorithm(DDPAlgorithm):
    config: DummyAlgoConfig

    def __init__(self, config: DummyAlgoConfig, policy: BasePolicy):
        super().__init__(config, policy)
        self.counter = 0
        self.global_step = 0
        self.break_action_chunk = config.break_action_chunk
        self.verbose = config.verbose
        self.verbose_feedback_interval = max(1, int(config.verbose_feedback_interval))
        self._stop_logged = False
        self._verbose_log(
            "initialized"
            f" | fake_infer={self.config.fake_inference_duration_sec:.3f}s"
            f" | fake_learn={self.config.fake_learn_duration_sec:.3f}s"
            f" | fake_learn_freq={self.config.fake_learn_freq}"
            f" | fake_total_steps={self.config.fake_total_steps}"
            f" | break_action_chunk={self.break_action_chunk}"
            f" | feedback_interval={self.verbose_feedback_interval}"
        )

    def _verbose_log(self, message: str) -> None:
        if self.verbose:
            logger.info(f"[DummyAlgorithm] {message}")

    def _should_log_progress(self, step: int) -> bool:
        return self.verbose and (
            step <= 1 or step % self.verbose_feedback_interval == 0
        )

    def infer(self, obs: dict) -> tuple[np.ndarray, InternalState]:
        next_step = self.global_step + 1
        batch_size = len(unbatch_aggregate(obs, aggregate_method="concat"))
        should_log = self._should_log_progress(next_step)
        start_time = time.perf_counter()
        if should_log:
            self._verbose_log(
                "infer start"
                f" | next_step={next_step}"
                f" | local_counter={self.counter}"
                f" | batch_size={batch_size}"
                f" | obs_keys={tuple(obs.keys())}"
                f" | sleep={self.config.fake_inference_duration_sec:.3f}s"
            )
        with torch.inference_mode():
            action, internal_state = self.policy.get_action_and_internal_state(obs)
        time.sleep(self.config.fake_inference_duration_sec)
        if should_log:
            self._verbose_log(
                "infer done"
                f" | next_step={next_step}"
                f" | action_shape={tuple(action.shape)}"
                f" | elapsed={time.perf_counter() - start_time:.3f}s"
            )
        return action, internal_state

    def feedback(
        self,
        *,
        obs: dict,
        internal_state: InternalState | None,
        terminated: bool,
        truncated: bool,
        next_obs: dict,
        reward: float,
        info: dict,
        next_terminated: bool,
        next_truncated: bool,
        prev_node: tuple,
    ) -> tuple:
        self.counter += 1
        self.global_step += 1
        should_learn = self.counter >= self.config.fake_learn_freq
        should_stop = self.global_step >= self.config.fake_total_steps
        steps_until_learn = max(0, self.config.fake_learn_freq - self.counter)
        if (
            self._should_log_progress(self.global_step)
            or next_terminated
            or next_truncated
            or should_learn
            or should_stop
        ):
            self._verbose_log(
                "feedback"
                f" | global_step={self.global_step}"
                f" | reward={reward:.4f}"
                f" | local_counter={self.counter}/{self.config.fake_learn_freq}"
                f" | steps_until_learn={steps_until_learn}"
                f" | next_terminated={next_terminated}"
                f" | next_truncated={next_truncated}"
                f" | should_learn={should_learn}"
                f" | should_stop={should_stop}"
            )
        return (-1, ""), self.global_step, {}

    def learn(self) -> tuple[int, dict]:
        self._verbose_log(
            "learn start"
            f" | global_step={self.global_step}"
            f" | buffered_steps={self.counter}"
            f" | sleep={self.config.fake_learn_duration_sec:.3f}s"
        )
        start_time = time.perf_counter()
        time.sleep(self.config.fake_learn_duration_sec)
        self.counter = 0
        self._verbose_log(
            "learn done"
            f" | global_step={self.global_step}"
            f" | elapsed={time.perf_counter() - start_time:.3f}s"
            f" | local_counter={self.counter}"
        )
        return self.global_step, {}

    def should_learn(self) -> bool:
        return self.counter >= self.config.fake_learn_freq

    def should_stop(self) -> bool:
        should_stop = self.global_step >= self.config.fake_total_steps
        if should_stop and not self._stop_logged:
            self._verbose_log(
                "stop condition reached"
                f" | global_step={self.global_step}"
                f" | fake_total_steps={self.config.fake_total_steps}"
            )
            self._stop_logged = True
        return should_stop

    def should_save(self) -> bool:
        return False

    def create_checkpoint(self) -> Checkpoint:
        self._verbose_log(f"create checkpoint | step={self.global_step}")
        return Checkpoint(step=self.global_step)

    def load_checkpoint(self, checkpoint: Checkpoint):
        self.global_step = checkpoint.step
        self.counter = 0
        self._stop_logged = False
        self._verbose_log(f"checkpoint loaded | step={self.global_step}")

    def activate_ddp(self, ddp_policy) -> None: ...

    def set_device(self, device) -> None: ...

    def get_serializable_buffer_data(self) -> dict:
        return {}

    def load_serializable_buffer_data(self, data: dict) -> None: ...

    def get_active_policy(self) -> BasePolicy:
        return self.policy

    def load_learner_state(self, checkpoint: Checkpoint) -> None:
        self.load_checkpoint(checkpoint)

    def get_server_data(self) -> tuple[int, dict, dict]:
        return self.global_step, {}, {}

    def load_server_data(self, global_step: int, meta_info: dict, data: dict) -> None:
        self.global_step = global_step
        self._stop_logged = False
        self._verbose_log(
            "server data loaded"
            f" | global_step={self.global_step}"
            f" | meta_keys={tuple(meta_info.keys())}"
            f" | data_keys={tuple(data.keys())}"
        )

    def create_ddp_checkpoint(self) -> Checkpoint:
        self._verbose_log(f"create ddp checkpoint | step={self.global_step}")
        return Checkpoint(step=self.global_step)
