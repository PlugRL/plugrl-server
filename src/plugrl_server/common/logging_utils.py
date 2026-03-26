import logging
import sys


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s|%(levelname)s|%(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
