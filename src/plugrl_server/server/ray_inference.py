from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from typing import Any

import numpy as np
import ray
import torch

from plugrl_server.algorithm.base_algorithm import BaseAlgoConfig
from plugrl_server.algorithm.registration import make_algo
from plugrl_server.common.checkpoint_manager import Checkpoint
from plugrl_server.common.data_utils import batch_aggregate
from plugrl_server.common.logging_utils import get_logger
from plugrl_server.common.metrics import MetricDict
from plugrl_server.policy.base_policy import BasePolicyConfig
from plugrl_server.policy.registration import make_policy
from plugrl_server.policy.state import slice_policy_step_state

logger = get_logger(__name__)


@dataclasses.dataclass
class InferenceWorkerSpec:
    algo_uid: str
    algo_config: BaseAlgoConfig
    policy_uid: str
    policy_config: BasePolicyConfig
    initial_checkpoint: Checkpoint | None = None


@ray.remote(num_gpus=1)
class RayInferenceWorker:
    def __init__(self, spec: InferenceWorkerSpec, worker_index: int):
        self._worker_index = worker_index
        self._requests_served = 0
        self._envs_served = 0
        self._batches_served = 0
        self._last_batch_size = 0
        self._last_duration = 0.0
        self._model_step = spec.initial_checkpoint.step if spec.initial_checkpoint else 0

        policy = make_policy(
            spec.policy_uid,
            config=dataclasses.replace(spec.policy_config, device=torch.device("cuda")),
        )
        self._algorithm = make_algo(
            spec.algo_uid,
            config=spec.algo_config,
            policy=policy,
        )
        if spec.initial_checkpoint is not None:
            self._load_model_checkpoint(spec.initial_checkpoint)

        logger.info(
            "Initialized Ray inference worker index=%s on device=%s",
            self._worker_index,
            getattr(policy, "device", "cuda"),
        )

    def describe_runtime(self) -> dict[str, Any]:
        device = getattr(self._algorithm.policy, "device", None)
        return dict(
            worker_index=self._worker_index,
            policy_device=str(device),
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()),
            current_cuda_device=(
                int(torch.cuda.current_device()) if torch.cuda.is_available() else None
            ),
            cuda_device_name=(
                str(torch.cuda.get_device_name(torch.cuda.current_device()))
                if torch.cuda.is_available()
                else None
            ),
            ray_gpu_ids=[float(gpu_id) for gpu_id in ray.get_gpu_ids()],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            model_step=int(self._model_step),
        )

    def infer_requests(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        started_at = time.perf_counter()
        obs = batch_aggregate(
            [request["obs"] for request in requests], aggregate_method="concat"
        )
        action, runtime_state = self._algorithm.infer(obs)
        step_state = self._algorithm.build_step_state_from_runtime_state(
            runtime_state,
            include_train_state=True,
        )

        responses: list[dict[str, Any]] = []
        start = 0
        total_envs = 0
        for request in requests:
            env_ids = np.asarray(request["env_ids"])
            req_size = len(env_ids)
            end = start + req_size
            responses.append(
                dict(
                    request_id=request["id"],
                    action=action[start:end],
                    step_states=[
                        slice_policy_step_state(step_state, slice(i, i + 1))
                        for i in range(start, end)
                    ],
                    env_ids=env_ids,
                    obs=request["obs"],
                    worker_index=self._worker_index,
                    model_step=self._model_step,
                )
            )
            start = end
            total_envs += req_size

        duration = time.perf_counter() - started_at
        self._requests_served += len(requests)
        self._envs_served += total_envs
        self._batches_served += 1
        self._last_batch_size = total_envs
        self._last_duration = duration
        return dict(
            worker_index=self._worker_index,
            duration=duration,
            num_requests=len(requests),
            num_envs=total_envs,
            model_step=self._model_step,
            responses=responses,
        )

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._load_model_checkpoint(checkpoint)
        self._model_step = checkpoint.step

    def get_metrics(self) -> dict[str, float]:
        return dict(
            requests_served=float(self._requests_served),
            envs_served=float(self._envs_served),
            batches_served=float(self._batches_served),
            last_batch_size=float(self._last_batch_size),
            last_duration=float(self._last_duration),
            model_step=float(self._model_step),
        )

    def _load_model_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._algorithm.global_step = checkpoint.step
        if checkpoint.model is not None:
            self._algorithm.policy.load_state_dict(checkpoint.model)


class RayInferenceWorkerGroup:
    def __init__(
        self,
        *,
        spec: InferenceWorkerSpec,
        num_workers: int,
    ) -> None:
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}.")
        self._workers = [
            RayInferenceWorker.remote(spec, worker_index=i)
            for i in range(num_workers)
        ]
        self._next_worker_index = 0
        self._route_key_to_worker_index: dict[str, int] = {}
        self._worker_index_to_route_keys: dict[int, set[str]] = {
            i: set() for i in range(num_workers)
        }
        self._route_key_to_last_env_ids: dict[str, tuple[int, ...]] = {}
        self._last_known_worker_metrics: dict[int, dict[str, float]] = {}

    def _partition_requests(
        self, requests: list[dict[str, Any]]
    ) -> list[tuple[int, list[dict[str, Any]]]]:
        shards: list[list[dict[str, Any]]] = [[] for _ in self._workers]
        for request in requests:
            route_key = request.get("route_key")
            if route_key is not None:
                route_key = str(route_key)
                worker_index = self._route_key_to_worker_index.get(route_key)
                if worker_index is None:
                    worker_index = self._next_worker_index % len(self._workers)
                    self._route_key_to_worker_index[route_key] = worker_index
                    self._worker_index_to_route_keys[worker_index].add(route_key)
                    self._next_worker_index += 1
                env_ids = request.get("env_ids")
                if env_ids is not None:
                    self._route_key_to_last_env_ids[route_key] = tuple(
                        int(env_id) for env_id in np.asarray(env_ids).tolist()
                    )
            else:
                worker_index = self._next_worker_index % len(self._workers)
                self._next_worker_index += 1
            shards[worker_index].append(request)
        return [
            (worker_index, shard)
            for worker_index, shard in enumerate(shards)
            if shard
        ]

    async def infer_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assignments = self._partition_requests(requests)
        refs = [
            self._workers[worker_index].infer_requests.remote(shard)
            for worker_index, shard in assignments
        ]
        shard_results = await asyncio.to_thread(ray.get, refs)

        responses_by_id: dict[str, dict[str, Any]] = {}
        for shard_result in shard_results:
            worker_index = int(shard_result["worker_index"])
            self._last_known_worker_metrics[worker_index] = dict(
                requests_served=float(
                    self._last_known_worker_metrics.get(worker_index, {}).get(
                        "requests_served", 0.0
                    )
                    + shard_result["num_requests"]
                ),
                envs_served=float(
                    self._last_known_worker_metrics.get(worker_index, {}).get(
                        "envs_served", 0.0
                    )
                    + shard_result["num_envs"]
                ),
                batches_served=float(
                    self._last_known_worker_metrics.get(worker_index, {}).get(
                        "batches_served", 0.0
                    )
                    + 1
                ),
                last_batch_size=float(shard_result["num_envs"]),
                last_duration=float(shard_result["duration"]),
                model_step=float(shard_result["model_step"]),
            )
            for response in shard_result["responses"]:
                responses_by_id[response["request_id"]] = response

        return [responses_by_id[request["id"]] for request in requests]

    async def sync_checkpoint(self, checkpoint: Checkpoint) -> None:
        refs = [worker.load_checkpoint.remote(checkpoint) for worker in self._workers]
        await asyncio.to_thread(ray.get, refs)
        for metrics in self._last_known_worker_metrics.values():
            metrics["model_step"] = float(checkpoint.step)

    async def refresh_metrics(self) -> None:
        refs = [worker.get_metrics.remote() for worker in self._workers]
        worker_metrics = await asyncio.to_thread(ray.get, refs)
        self._last_known_worker_metrics = {
            worker_index: metrics
            for worker_index, metrics in enumerate(worker_metrics)
        }

    async def describe_runtimes(self) -> list[dict[str, Any]]:
        refs = [worker.describe_runtime.remote() for worker in self._workers]
        return await asyncio.to_thread(ray.get, refs)

    def as_metrics(self) -> MetricDict:
        nested: dict[str, dict[str, float]] = {}
        for worker_index, metrics in sorted(self._last_known_worker_metrics.items()):
            nested[f"worker_{worker_index:02d}"] = metrics
        return dict(inference_workers=nested)

    def describe_route_assignments(self) -> dict[int, list[dict[str, Any]]]:
        assignments: dict[int, list[dict[str, Any]]] = {}
        for worker_index, route_keys in sorted(self._worker_index_to_route_keys.items()):
            entries: list[dict[str, Any]] = []
            for route_key in sorted(route_keys):
                entries.append(
                    dict(
                        route_key=route_key,
                        last_env_ids=list(self._route_key_to_last_env_ids.get(route_key, ())),
                    )
                )
            assignments[worker_index] = entries
        return assignments

    def describe_route_assignment(
        self, route_key: str | None
    ) -> dict[str, Any] | None:
        if route_key is None:
            return None
        normalized_route_key = str(route_key)
        worker_index = self._route_key_to_worker_index.get(normalized_route_key)
        if worker_index is None:
            return None
        return dict(
            route_key=normalized_route_key,
            worker_index=worker_index,
            last_env_ids=list(
                self._route_key_to_last_env_ids.get(normalized_route_key, ())
            ),
        )

    @property
    def num_workers(self) -> int:
        return len(self._workers)
