from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)

import plugrl_server.algorithm.dummy_algorithm
import plugrl_server.algorithm.evaluation

try:
    import plugrl_server.algorithm.dppo.dppo_config  # noqa: F401
    import plugrl_server.algorithm.dppo.dppo_dist_config  # noqa: F401
except Exception as e:
    logger.warning(f"Could not import DPPO algorithm module for reason: {e}")
