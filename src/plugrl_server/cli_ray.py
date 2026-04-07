import os

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
import dataclasses
import asyncio
from typing import Literal

import ray
import torch

import plugrl_server
from plugrl_server.cli import (
    Args as BaseArgs,
    build_cli_from_registry,
)
from plugrl_server.policy.registration import make_policy
from plugrl_server.algorithm.registration import make_algo
from plugrl_server.common.logging_utils import configure_logging, get_logger
from plugrl_server.common.metrics import (
    close_metric_sink_and_tracker,
    init_metric_sink_by_tracker,
)
from plugrl_server.server.ray_agent_server import RayAgentServer
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.server.ray_inference import (
    InferenceWorkerSpec,
    RayInferenceWorkerGroup,
)

logger = get_logger(__name__)


@dataclasses.dataclass
class RayArgs(BaseArgs):
    num_infer_workers: int | None = None
    local_policy_device: Literal["cpu", "cuda"] = "cpu"


def cli() -> RayArgs:
    return build_cli_from_registry(RayArgs)


def _main(args: RayArgs):
    configure_logging(args.log_level)

    logger.info(f"plugrl_server version: {plugrl_server.__version__}")
    logger.info(f"Algorithm: {args.algo_uid}, Config: {args.algo}")
    logger.info(f"Policy: {args.policy_uid}, Config: {args.policy}")

    if not ray.is_initialized():
        ray.init()
        logger.info("Initialized Ray.")
    logger.info("Ray cluster resources: %s", ray.cluster_resources())
    logger.info("Ray available resources: %s", ray.available_resources())

    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir,
        config=dataclasses.asdict(args),
        overwrite=args.overwrite,
        resume=args.resume,
    )
    logger.info(
        f"Checkpoint Manager created: \n{checkpoint_manager} at {args.checkpoint_dir}"
    )

    metric_sink, tracker = init_metric_sink_by_tracker(
        args, resuming=args.resume, log_code=not args.resume, enabled=args.track.enabled
    )

    policy = make_policy(
        args.policy_uid,
        config=dataclasses.replace(args.policy, device=args.local_policy_device),
    )
    logger.info(
        "Local training policy created on device=%s",
        args.local_policy_device,
    )
    if args.local_policy_device == "cuda":
        logger.warning(
            "Local training policy is running on CUDA. In Ray mode the main "
            "process still performs DPPO learning locally, so its GPU memory "
            "usage adds to the Ray inference workers' GPU usage."
        )

    algo = make_algo(args.algo_uid, config=args.algo, policy=policy)
    algo.init_optimizers()
    logger.info(f"Algorithm created: \n{algo}")

    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found in {args.checkpoint_dir} to resume."
            )
        logger.info(f"Resumed from checkpoint at step {checkpoint.step}")
        algo.load_checkpoint(checkpoint)
    else:
        checkpoint = algo.create_checkpoint()

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("Ray inference mode requires at least one CUDA device.")
    num_infer_workers = args.num_infer_workers or available_gpus
    if num_infer_workers > available_gpus:
        raise ValueError(
            f"Requested {num_infer_workers} inference workers, but only {available_gpus} CUDA devices are visible."
        )

    worker_spec = InferenceWorkerSpec(
        algo_uid=args.algo_uid,
        algo_config=args.algo,
        policy_uid=args.policy_uid,
        policy_config=args.policy,
        initial_checkpoint=checkpoint,
    )
    inference_workers = RayInferenceWorkerGroup(
        spec=worker_spec,
        num_workers=num_infer_workers,
    )
    logger.info("Initialized %s Ray inference worker(s).", num_infer_workers)
    worker_runtimes = asyncio.run(inference_workers.describe_runtimes())
    for runtime in worker_runtimes:
        logger.info("Ray inference worker runtime: %s", runtime)
    if not all(runtime.get("cuda_available") for runtime in worker_runtimes):
        raise RuntimeError(
            f"At least one Ray inference worker does not have CUDA available: {worker_runtimes}"
        )
    if not all(runtime.get("ray_gpu_ids") for runtime in worker_runtimes):
        raise RuntimeError(
            f"At least one Ray inference worker was not assigned a GPU by Ray: {worker_runtimes}"
        )

    server = RayAgentServer(
        algorithm=algo,
        checkpoint_manager=checkpoint_manager,
        metric_sink=metric_sink,
        inference_workers=inference_workers,
        host=args.host,
        port=args.port,
        mini_infer_batch_size=args.mini_infer_batch_size,
        show_metric_table=args.show_metric_table,
        show_progress_bar=args.show_progress_bar,
        rollout_only=args.rollout_only,
    )
    try:
        server.serve_forever()
    finally:
        close_metric_sink_and_tracker(metric_sink, tracker)


def main():
    _main(cli())


if __name__ == "__main__":
    main()
