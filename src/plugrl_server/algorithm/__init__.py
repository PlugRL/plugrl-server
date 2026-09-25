import plugrl_server.algorithm.fpo.fpo
import plugrl_server.algorithm.fpo.fpo_config
import plugrl_server.algorithm.dummy_algorithm
import plugrl_server.algorithm.evaluation

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)


# The classes, not only their configs. Importing the config modules alone
# registered `dppo` and `dppo-dist` in the CLI menu while leaving the classes
# unregistered, so selecting either parsed, started up, printed its config and
# died with `KeyError: 'Algorithm dppo is not registered.'`.
import plugrl_server.algorithm.dppo.dppo  # noqa: F401,E402
import plugrl_server.algorithm.dppo.dppo_config  # noqa: F401,E402
import plugrl_server.algorithm.dppo.dppo_dist  # noqa: F401,E402
import plugrl_server.algorithm.dppo.dppo_dist_config  # noqa: F401,E402
