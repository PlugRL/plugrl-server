import asyncio
from typing import Any

import ray
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from plugrl_server.algorithm.base_algorithm import BaseAlgorithm, DDPAlgorithm
from plugrl_server.common.checkpoint_manager import CheckpointManager, Checkpoint


class LocalTrainingBackend:
    def __init__(
        self,
        *,
        algorithm: BaseAlgorithm,
        checkpoint_manager: CheckpointManager,
        writer: SummaryWriter,
    ) -> None:
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._writer = writer

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    async def process_learn(self) -> None:
        self._algorithm.pre_learn()
        step, log_dict = await asyncio.to_thread(self._algorithm.learn)
        self._algorithm.post_learn()
        self.log(log_dict, step=step)

    async def process_save(self) -> None:
        checkpoint = self._algorithm.create_checkpoint()
        await asyncio.to_thread(self._checkpoint_manager.save_checkpoint, checkpoint)

    def log(self, log_dict: dict, *, step: int) -> None:
        for key, value in log_dict.items():
            self._writer.add_scalar(key, value, step)


class RayTrainingBackend:
    def __init__(
        self,
        *,
        algorithm: DDPAlgorithm,
        checkpoint_manager: CheckpointManager,
        writer: SummaryWriter,
        learner_actor_ref: ray.ObjectRef,
    ) -> None:
        self._algorithm = algorithm
        self._checkpoint_manager = checkpoint_manager
        self._writer = writer
        self._learner_actor: Any = learner_actor_ref

    def should_learn(self) -> bool:
        return self._algorithm.should_learn()

    def should_save(self) -> bool:
        return self._algorithm.should_save()

    def should_stop(self) -> bool:
        return self._algorithm.should_stop()

    async def process_learn(self) -> None:
        logger.info("Scheduler initiating distributed training via LearnerActor.")
        self._algorithm.pre_learn()
        global_step, meta_info, serializable_buffer_data = (
            self._algorithm.get_server_data()
        )
        buffer_data_ref = await asyncio.to_thread(ray.put, serializable_buffer_data)

        learn_ref = self._learner_actor.learn.remote(
            global_step, meta_info, buffer_data_ref
        )
        checkpoint, global_step, train_info = await asyncio.to_thread(ray.get, learn_ref)

        await self._update_inference_policy(checkpoint)
        await asyncio.to_thread(self.log, train_info, step=global_step)
        logger.info(f"Logged training info for step {global_step}.")
        self._algorithm.post_learn()

    async def process_save(self) -> None:
        checkpoint = await asyncio.to_thread(self._algorithm.create_ddp_checkpoint)
        await asyncio.to_thread(self._checkpoint_manager.save_checkpoint, checkpoint)
        logger.info(f"Checkpoint saved locally at step {checkpoint.step}.")

    async def shutdown(self) -> None:
        if ray.is_initialized():
            await asyncio.to_thread(ray.shutdown)

    async def _update_inference_policy(self, checkpoint: Checkpoint) -> None:
        self._algorithm.load_learner_state(checkpoint)
        logger.info("Local inference policy successfully updated.")

    def log(self, log_dict: dict, *, step: int) -> None:
        for key, value in log_dict.items():
            self._writer.add_scalar(key, value, step)
