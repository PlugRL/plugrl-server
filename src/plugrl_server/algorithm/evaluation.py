import dataclasses
import pathlib
import numpy as np
import torch

from .base_algorithm import BaseAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from plugrl_server.common.metrics import MetricDict
from plugrl_server.policy.base_policy import BasePolicy
from plugrl_server.policy.state import PolicyRuntimeState, PolicyTrainState
from plugrl_server.common.checkpoint_manager import (
    Checkpoint,
    load_checkpoint_from_path,
)
from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)

UID = "eval"


@register_algo_config(UID)
@dataclasses.dataclass
class EvalConfig(BaseAlgoConfig):
    break_action_chunk: bool = False
    policy_checkpoint_path: pathlib.Path | None = None


@register_algo(UID)
class Evaluation(BaseAlgorithm):
    config: EvalConfig

    def __init__(self, config: EvalConfig, policy: BasePolicy):
        super().__init__(config, policy)
        self.counter = 0
        self.break_action_chunk = config.break_action_chunk
        if config.policy_checkpoint_path is not None:
            logger.info(
                f"Loading policy checkpoint from {config.policy_checkpoint_path}"
            )
            checkpoint_manager = load_checkpoint_from_path(
                config.policy_checkpoint_path
            )
            self.load_checkpoint(checkpoint_manager)

    def infer(self, obs: dict) -> tuple[np.ndarray, PolicyRuntimeState]:
        with torch.inference_mode():
            action, runtime_state = self.policy.get_action_and_runtime_state(obs)
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
        log_dict = {}
        if next_terminated or next_truncated:
            if "episode" in info:
                log_dict = dict(
                    episode=dict(
                        reward=info["episode"]["r"],
                        length=info["episode"]["l"],
                        success=info["episode"]["s"],
                    )
                )
                self.record_episode_metrics(info["episode"])
        return (-1, ""), 0, log_dict

    def learn_impl(self) -> tuple[int, MetricDict]:
        self.counter = 0
        return 0, {}

    def should_learn(self) -> bool:
        return False

    def should_stop(self) -> bool:
        return False

    def should_save(self) -> bool:
        return False

    def create_checkpoint(self) -> Checkpoint:
        return Checkpoint(step=0)

    def load_checkpoint(self, checkpoint: Checkpoint):
        if checkpoint.model is not None:
            self.policy.load_state_dict(checkpoint.model)
