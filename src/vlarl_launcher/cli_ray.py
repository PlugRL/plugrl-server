import dataclasses
import torch
import sys
from typing import Literal
import pathlib
import tyro
import uuid
import datetime
import dateutil.tz
import ray
from loguru import logger

import vlarl_launcher
from vlarl_launcher.algorithm.base_algorithm import BaseAlgoConfig
from vlarl_launcher.policy.base_policy import BasePolicyConfig
from vlarl_launcher.policy.registration import REGISTERED_POLICY_CONFIGS, make_policy
from vlarl_launcher.algorithm.registration import REGISTERED_ALGO_CONFIGS, make_algo
from vlarl_launcher.server.ray_agent_server import RayAgentServer
from vlarl_launcher.common.checkpoint_manager import CheckpointManager
from vlarl_launcher.server.ray_learner import LearnerActor

@dataclasses.dataclass
class TrackArgs:
    project_name: str = "VLARL"
    entity: str = ""
    tracker: Literal["wandb", "swanlab"] = "swanlab"
    enabled: bool = False
    
@dataclasses.dataclass
class Args:
    algo_uid: tyro.conf._markers.Suppress[str]
    algo: BaseAlgoConfig
    
    policy_uid: tyro.conf._markers.Suppress[str]
    policy: BasePolicyConfig

    log_level: Literal["debug", "info"] = "info"

    host: str = "0.0.0.0"
    port: int = 8000

    prefix: str | None = None
    suffix: str | None = None
    exp_name: str | None = None
    overwrite: bool = False
    resume: bool = False
    track: TrackArgs = dataclasses.field(default_factory=TrackArgs)
    
    infer_gpu: int | None = None
    ddp_gpus: list[int] | None = None
    master_addr: str | None = None
    master_port: str | None = None
    
    checkpoint_base_dir: str = "./checkpoints"
    
    def __post_init__(self):
        if self.exp_name is None:
            if self.prefix is None:
                self.prefix = str(uuid.uuid4().fields[-1])[:5]

            self.exp_name = create_exp_name(self.prefix)
            if self.suffix:
                self.exp_name = f"{self.exp_name}_{self.suffix}"
        
        self.policy.device = "cpu"  # Ensure policy is on CPU initially

    @property
    def checkpoint_dir(self):
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.full_exp_name).resolve()
    
    @property
    def full_exp_name(self):
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return f"{self.algo_uid}/{self.policy_uid}/{self.exp_name}"
                
def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (timestamp, exp_prefix, exp_id, seed)
    
_CONFIGS_DICT = {}
for policy_uid, policy_cfg in REGISTERED_POLICY_CONFIGS.items():
    support_algos = policy_cfg.supported_algos if policy_cfg.supported_algos is not None else [(k, "default") for k in REGISTERED_ALGO_CONFIGS.keys()]
    for algo_uid, variant_uid in support_algos:
        if algo_uid not in REGISTERED_ALGO_CONFIGS or variant_uid not in REGISTERED_ALGO_CONFIGS[algo_uid]:
            logger.warning(f"Algorithm {algo_uid} with variant {variant_uid} is not registered, skipping...")
            continue
        algo_cfg = REGISTERED_ALGO_CONFIGS[algo_uid][variant_uid]
        key = f"{algo_uid}-{variant_uid}/{policy_uid}".lower()
        policy_cfg.algo = algo_uid
        _CONFIGS_DICT[key] = Args(
            algo_uid=algo_uid,
            algo=algo_cfg,
            policy_uid=policy_uid,
            policy=policy_cfg,
        )

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def init_tracker(args: Args, *, resuming: bool, log_code: bool, enabled: bool = True):
    if args.track.tracker == "wandb":
        import wandb
        tracker_module = wandb
    else:
        import swanlab
        tracker_module = swanlab

    if not enabled:
        tracker = tracker_module.init(mode="disabled")
        return tracker

    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "run_id.txt").read_text().strip()
        tracker = tracker_module.init(id=run_id, resume="must", project=args.track.project_name, entity=args.track.entity or None)
    else:
        tracker = tracker_module.init(
            entity=args.track.entity or None,
            project=args.track.project_name,
            name=args.full_exp_name,
            config=dataclasses.asdict(args),
            save_code=log_code,
        )
        (ckpt_dir / "run_id.txt").write_text(tracker.id)
            
    return tracker

def _main(args: Args):
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