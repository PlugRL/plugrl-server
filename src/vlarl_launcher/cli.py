import dataclasses
import sys
from typing import Literal
import tyro
from loguru import logger

import vlarl_launcher
from vlarl_launcher.algorithm.base import BaseAlgoConfig
from vlarl_launcher.policy.base_policy import BasePolicyConfig
from vlarl_launcher.policy.registration import REGISTERED_POLICY_CONFIGS, make_policy
from vlarl_launcher.algorithm.registration import REGISTERED_ALGO_CONFIGS, make_algo
from vlarl_launcher.server.websocket_agent_server import WebSocketAgentServer


@dataclasses.dataclass
class Args:
    algo_uid: tyro.conf._markers.Suppress[str]
    algo: BaseAlgoConfig
    
    policy_uid: tyro.conf._markers.Suppress[str]
    policy: BasePolicyConfig
    
    log_level: Literal["DEBUG", "INFO"] = "INFO"
    
    host: str = "0.0.0.0"
    port: int = 8000
    
# _CONFIGS_DICT = {k.lower(): Args(algo_uid=k, algo=v) for k, v in REGISTERED_ALGO_CONFIGS.items()}
_CONFIGS_DICT = {}
for policy_uid, policy_cfg in REGISTERED_POLICY_CONFIGS.items():
    support_algos = policy_cfg.supported_algos if policy_cfg.supported_algos is not None else REGISTERED_ALGO_CONFIGS.keys()
    for algo_uid in support_algos:
        if algo_uid not in REGISTERED_ALGO_CONFIGS:
            continue
        algo_cfg = REGISTERED_ALGO_CONFIGS[algo_uid]
        key = f"{algo_uid}/{policy_uid}".lower()
        _CONFIGS_DICT[key] = Args(
            algo_uid=algo_uid,
            algo=algo_cfg,
            policy_uid=policy_uid,
            policy=policy_cfg,
        )

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level}])
    
    logger.info(f"vlarl_launcher version: {vlarl_launcher.__version__}")
    logger.info(f"Algorithm: {args.algo_uid}, Config: {args.algo}")
    logger.info(f"Policy: {args.policy_uid}, Config: {args.policy}")
    
    policy = make_policy(args.policy_uid, config=args.policy)
    logger.info(f"Policy created: {policy}")
    
    algo = make_algo(args.algo_uid, policy=policy, config=args.algo)
    logger.info(f"Algorithm created: {algo}")

    server = WebSocketAgentServer(algo, host=args.host, port=args.port)
    server.serve_forever()
    
def main():
    _main(cli())
    