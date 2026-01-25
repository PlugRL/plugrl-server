import plugrl_server.algorithm.dummy_algorithm
import plugrl_server.algorithm.evaluation

try:
    import plugrl_server.algorithm.dppo.dppo_config  # noqa: F401
    import plugrl_server.algorithm.dppo.dppo_dist_config  # noqa: F401
except Exception as e:
    print(f"Could not import DPPO algorithm module for reason: {e}")
