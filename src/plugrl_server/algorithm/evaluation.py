import dataclasses
import numpy as np
import pathlib
import torch
import tqdm
from collections import deque
from loguru import logger

from .base_algorithm import BaseAlgorithm, BaseAlgoConfig
from .registration import register_algo, register_algo_config

from plugrl_server.policy.base_policy import InternalState, BasePolicy
from plugrl_server.policy.state import PolicyRuntimeState
from plugrl_server.common.checkpoint_manager import (
    Checkpoint,
    load_checkpoint_from_path,
)

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
        self._eval_pbar = tqdm.tqdm(
            desc="Eval",
            unit="episode",
            leave=False,
            ncols=0,
            smoothing=0.01,
        )
        self._record_episode_stats = deque()
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
            action, internal_state = self.policy.get_action_and_internal_state(obs)
        return action, internal_state

    def feedback(
        self,
        *,
        obs: dict,
        internal_state: PolicyRuntimeState,
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
                log_dict = {
                    "episode/reward": info["episode"]["r"],
                    "episode/length": info["episode"]["l"],
                    "episode/success": info["episode"]["s"],
                }
                self._record_episode_stats.append(log_dict)
                self._eval_pbar.set_postfix(self.get_recorded_episode_stats())
                self._eval_pbar.update(1)
        return (-1, ""), 0, log_dict

    def learn(self) -> tuple[int, dict]:
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
            self.active_policy.load_state_dict(checkpoint.model)

    def get_active_policy(self) -> BasePolicy:
        return self.policy

    def get_recorded_episode_stats(self) -> dict:
        avg_stats = {}
        if len(self._record_episode_stats) > 0:
            keys = self._record_episode_stats[0].keys()
            for key in keys:
                avg_stats[key] = np.mean(
                    [ep_stats[key] for ep_stats in self._record_episode_stats]
                )
        return avg_stats
