import dataclasses
import sys

import ray
import torch
from loguru import logger

import vlarl_launcher
from vlarl_launcher.cli import Args as BaseArgs, build_cli_from_registry, init_tracker
from vlarl_launcher.policy.registration import make_policy
from vlarl_launcher.algorithm.registration import make_algo
from vlarl_launcher.server.ray_agent_server import RayAgentServer
from vlarl_launcher.common.checkpoint_manager import CheckpointManager
from vlarl_launcher.server.ray_learner import LearnerActor


@dataclasses.dataclass
class RayArgs(BaseArgs):
    infer_gpu: int | None = None
    ddp_gpus: list[int] | None = None
    master_addr: str | None = None
    master_port: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.policy.device = "cpu"  # Ensure policy is on CPU initially


def cli() -> RayArgs:
    return build_cli_from_registry(RayArgs)


def _main(args: RayArgs):
    logger.configure(handlers=[{"sink": sys.stderr, "level": args.log_level.upper()}])
            
    logger.info(f"vlarl_launcher version: {vlarl_launcher.__version__}")
    logger.info(f"Algorithm: {args.algo_uid}, Config: {args.algo}")
    logger.info(f"Policy: {args.policy_uid}, Config: {args.policy}")

    if not ray.is_initialized():
        ray.init()
        logger.info("Initialized Ray.")
    
    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir, config=dataclasses.asdict(args), overwrite=args.overwrite, resume=args.resume
    )   
    logger.info(f"Checkpoint Manager created: \n{checkpoint_manager} at {args.checkpoint_dir}")

    tracker = init_tracker(args, resuming=args.resume, log_code=not args.resume, enabled=args.track.enabled)
    
    policy = make_policy(args.policy_uid, config=args.policy)
    algo = make_algo(args.algo_uid, config=args.algo, policy=policy)
    assert isinstance(algo, vlarl_launcher.algorithm.base_algorithm.DDPAlgorithm)

    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir} to resume.")
        logger.info(f"Resumed from checkpoint at step {checkpoint.step}")
        algo.load_checkpoint(checkpoint)
    
    algo_ref = ray.put(algo)
    if args.ddp_gpus is None:
        ddp_gpus = list(range(torch.cuda.device_count()))
        logger.info(f"No DDP GPUs specified, using all available GPUs: {ddp_gpus}")
    else:
        ddp_gpus = args.ddp_gpus
    learner_ref = LearnerActor.remote([algo_ref], ddp_gpus=ddp_gpus, master_addr=args.master_addr, master_port=args.master_port)
    
    inference_device = torch.device("cuda") if args.infer_gpu is None else torch.device(f"cuda:{args.infer_gpu}")
    algo.set_device(inference_device)
    server = RayAgentServer(algo, checkpoint_manager, tracker, learner_ref, host=args.host, port=args.port)
    server.serve_forever()
    
def main():
    _main(cli())