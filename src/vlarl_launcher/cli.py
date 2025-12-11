import dataclasses
import sys
from typing import Literal, TypeVar
import pathlib
import tyro
import uuid
import datetime
import dateutil.tz
from loguru import logger
import functools

import vlarl_launcher
from vlarl_launcher.algorithm.base_algorithm import BaseAlgoConfig
from vlarl_launcher.policy.base_policy import BasePolicyConfig
from vlarl_launcher.policy.registration import REGISTERED_POLICY_CONFIGS, make_policy
from vlarl_launcher.algorithm.registration import REGISTERED_ALGO_CONFIGS, make_algo
from vlarl_launcher.server.websocket_agent_server import WebSocketAgentServer
from vlarl_launcher.common.checkpoint_manager import CheckpointManager
from vlarl_launcher.common.tyro_utils import subcommand_cli_from_nested_dict

@dataclasses.dataclass
class TrackArgs:
    project_name: str = "VLARL"
    entity: str = ""
    tracker: Literal["wandb", "swanlab"] = "swanlab"
    enabled: bool = False
    
@dataclasses.dataclass
class Args:
    track: TrackArgs
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
    
    checkpoint_base_dir: str = "./checkpoints"
    
    def __post_init__(self):
        if self.exp_name is None:
            if self.prefix is None:
                self.prefix = str(uuid.uuid4().fields[-1])[:5]

            self.exp_name = create_exp_name(self.prefix)
            if self.suffix is not None:
                self.exp_name = f"{self.exp_name}_{self.suffix}"
        
        self.policy.algo = self.algo_uid
                
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

ArgsT = TypeVar("ArgsT", bound="Args")

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

def cli() -> Args:
    return build_cli_from_registry(Args)


def build_cli_from_registry(args_cls: type[ArgsT]) -> ArgsT:
    configs = {}
    for policy_uid, policy_variants in REGISTERED_POLICY_CONFIGS.items():
        configs[policy_uid] = {}
        for policy_variant, policy_config in policy_variants.items():
            configs[policy_uid][policy_variant] = {}
            for algo_uid, algo_variants in REGISTERED_ALGO_CONFIGS.items():
                configs[policy_uid][policy_variant][algo_uid] = {}
                for algo_variant, algo_config in algo_variants.items():
                    configs[policy_uid][policy_variant][algo_uid][algo_variant] = functools.partial(
                        args_cls,
                        policy_uid=policy_uid,
                        policy=policy_config,
                        algo_uid=algo_uid,
                        algo=algo_config,
                    )
    return subcommand_cli_from_nested_dict(configs)[0][0][0]
        

def init_writer_by_tracker(args: Args, *, resuming: bool, log_code: bool, enabled: bool = True):
    if args.track.tracker == "wandb":
        import wandb
        tracker_module = wandb
    else:
        import swanlab
        tracker_module = swanlab
        swanlab.sync_tensorboard_torch()
            
    if not enabled:
        tracker = tracker_module.init(mode="disabled", sync_tensorboard=True)
    else:
        ckpt_dir = args.checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        project_name = f"{args.track.project_name}-{args.algo_uid}-{args.policy_uid}"
        if resuming:
            run_id = (ckpt_dir / "run_id.txt").read_text().strip()
            tracker = tracker_module.init(id=run_id, resume="must", project=project_name, entity=args.track.entity or None, sync_tensorboard=True)
        else:
            tracker = tracker_module.init(
                entity=args.track.entity or None,
                project=project_name,
                name=args.full_exp_name,
                config=dataclasses.asdict(args),
                save_code=log_code,
                sync_tensorboard=True,
            )
            run_id_value = tracker.id
            if run_id_value is None:
                raise RuntimeError("Tracker did not return a run id")
            (ckpt_dir / "run_id.txt").write_text(run_id_value)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.checkpoint_dir / "tensorboard"))
    return writer, tracker

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level.upper(), "format": "{time:HH:mm:ss}|{level}|{message}"}])
            
    logger.info(f"vlarl_launcher version: {vlarl_launcher.__version__}")
    logger.info(f"Algorithm: {args.algo_uid}, Config: {args.algo}")
    logger.info(f"Policy: {args.policy_uid}, Config: {args.policy}")
    
    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir, config=dataclasses.asdict(args), overwrite=args.overwrite, resume=args.resume
    )   
    logger.info(f"Checkpoint Manager created: \n{checkpoint_manager} at {args.checkpoint_dir}")

    writer, tracker = init_writer_by_tracker(args, resuming=args.resume, log_code=not args.resume, enabled=args.track.enabled)
    
    policy = make_policy(args.policy_uid, config=args.policy)
    logger.info(f"Policy created...")
    
    algo = make_algo(args.algo_uid, policy=policy, config=args.algo)
    algo.init_optimizers()
    logger.info(f"Algorithm created: \n{algo}")

    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir} to resume.")
        logger.info(f"Resumed from checkpoint at step {checkpoint.step}")
        algo.load_checkpoint(checkpoint)

    server = WebSocketAgentServer(algo, checkpoint_manager, writer, host=args.host, port=args.port)
    server.serve_forever()
    
def main():
    _main(cli())
    