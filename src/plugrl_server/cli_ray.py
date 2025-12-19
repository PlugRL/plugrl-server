import os
os.environ.setdefault("RAY_DISABLE_METRICS", "1")
import dataclasses
import sys

import ray
import torch
from loguru import logger

import plugrl_server
from plugrl_server.cli import Args as BaseArgs, build_cli_from_registry, init_writer_by_tracker
from plugrl_server.policy.registration import make_policy
from plugrl_server.algorithm.registration import make_algo
from plugrl_server.server.ray_agent_server import RayAgentServer
from plugrl_server.common.checkpoint_manager import CheckpointManager
from plugrl_server.server.ray_learner import LearnerActor, LearnerAlgoSpec


@dataclasses.dataclass
class RayArgs(BaseArgs):
    infer_gpu: int | None = None
    num_ddp_gpus: int | None = None
    master_addr: str | None = None
    master_port: str | None = None


def cli() -> RayArgs:
    return build_cli_from_registry(RayArgs)


def _main(args: RayArgs):
    logger.configure(handlers=[{"sink": sys.stderr, "level": args.log_level.upper()}])
            
    logger.info(f"plugrl_server version: {plugrl_server.__version__}")
    logger.info(f"Algorithm: {args.algo_uid}, Config: {args.algo}")
    logger.info(f"Policy: {args.policy_uid}, Config: {args.policy}")

    if not ray.is_initialized():
        ray.init()
        logger.info("Initialized Ray.")
    
    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir, config=dataclasses.asdict(args), overwrite=args.overwrite, resume=args.resume
    )   
    logger.info(f"Checkpoint Manager created: \n{checkpoint_manager} at {args.checkpoint_dir}")

    writer, tracker = init_writer_by_tracker(args, resuming=args.resume, log_code=not args.resume, enabled=args.track.enabled)
    
    policy = make_policy(args.policy_uid, config=dataclasses.replace(args.policy, device=torch.device("cuda", args.infer_gpu) if args.infer_gpu is not None else "cuda"))
    algo = make_algo(args.algo_uid, config=args.algo, policy=policy)
    assert isinstance(algo, plugrl_server.algorithm.base_algorithm.DDPAlgorithm), f"Algorithm {args.algo_uid} is not a DDPAlgorithm."
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir} to resume.")
        logger.info(f"Resumed from checkpoint at step {checkpoint.step}")
        algo.load_learner_state(checkpoint)
    
    learner_spec = LearnerAlgoSpec(
        algo_uid=args.algo_uid,
        algo_config=args.algo,
        policy_uid=args.policy_uid,
        policy_config=args.policy,
        initial_checkpoint=algo.create_ddp_checkpoint() if args.resume else None,
    )
    if args.num_ddp_gpus is None:
        ddp_gpus = list(range(torch.cuda.device_count()))
        logger.info(f"No DDP GPUs specified, using all available GPUs: {ddp_gpus}")
    else:
        ddp_gpus = list(range(args.num_ddp_gpus))
    learner_ref = LearnerActor.remote(learner_spec, ddp_gpus=ddp_gpus, master_addr=args.master_addr, master_port=args.master_port)
    
    server = RayAgentServer(algo, checkpoint_manager, writer, learner_ref, host=args.host, port=args.port)
    server.serve_forever()
    
def main():
    _main(cli())