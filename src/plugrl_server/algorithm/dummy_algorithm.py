import dataclasses
import numpy as np
import time
import torch

from .base_algorithm import BaseAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from plugrl_server.policy.base_policy import BasePolicy
from plugrl_server.policy.state import PolicyRuntimeState, PolicyTrainState
from plugrl_server.common.checkpoint_manager import Checkpoint

UID = "dummy"


@register_algo_config(UID)
@dataclasses.dataclass
class DummyAlgoConfig(BaseAlgoConfig):
    fake_inference_duration_sec: float = 0.1
    fake_learn_duration_sec: float = 10.0
    fake_learn_freq: int = 100
    global_steps: int | None = 300

    break_action_chunk: bool = False


@register_algo(UID)
class DummyAlgorithm(BaseAlgorithm):
    config: DummyAlgoConfig

    def __init__(self, config: DummyAlgoConfig, policy: BasePolicy):
        super().__init__(config, policy)
        self.counter = 0
        self.global_step = 0
        self.break_action_chunk = config.break_action_chunk

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            action, runtime_state = self.policy.get_action_and_runtime_state(obs)
        time.sleep(self.config.fake_inference_duration_sec)
        return action, runtime_state

    def feedback(
        self,
        *,
        obs: dict,
        runtime_state: PolicyRuntimeState,
        train_state: PolicyTrainState = None,
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
        return (-1, ""), self.global_step, {}

    def learn_impl(self) -> tuple[int, dict[str, float]]:
        time.sleep(self.config.fake_learn_duration_sec)
        self.counter = 0
        return self.global_step, {}

    def should_learn(self) -> bool:
        return self.counter >= self.config.fake_learn_freq

    def should_stop(self) -> bool:
        assert self.config.global_steps is not None
        return self.global_step >= self.config.global_steps

    def should_save(self) -> bool:
        return False

    def create_checkpoint(self) -> Checkpoint:
        return Checkpoint(step=self.global_step)

    def load_checkpoint(self, checkpoint: Checkpoint):
        self.global_step = checkpoint.step
        self.counter = 0
