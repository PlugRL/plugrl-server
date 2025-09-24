import dataclasses
import sys
from typing import Literal
import tyro
from loguru import logger

import vlarl_launcher
from vlarl_launcher.algorithm.base import BaseAlgoConfig
from vlarl_launcher.common.registration import REGISTERED_ALGO_CONFIGS, make_algo
from vlarl_launcher.server.websocket_agent_server import WebSocketAgentServer


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    algo: BaseAlgoConfig
    
    log_level: Literal["DEBUG", "INFO"] = "INFO"
    
    host: str = "0.0.0.0"
    port: int = 8000
    
_CONFIGS_DICT = {k.lower(): Args(uid=k, algo=v) for k, v in REGISTERED_ALGO_CONFIGS.items()}

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level}])
    
    logger.info(f"vlarl_launcher version: {vlarl_launcher.__version__}")
    logger.info(f"Algorithm: {args.uid}, Config: {args.algo}")
    
    algo = make_algo(args.uid, config=args.algo)
    
    server = WebSocketAgentServer(algo, host=args.host, port=args.port)
    server.serve_forever()
    
def main():
    _main(cli())
    