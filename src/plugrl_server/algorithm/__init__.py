import plugrl_server.algorithm.dummy_algorithm
import plugrl_server.algorithm.evaluation

try:
    import plugrl_server.algorithm.dppo.dppo_config
    import plugrl_server.algorithm.dppo.dppo_dist_config
except Exception:
    print(f"Could not import DPPO algorithm module for reason: {Exception}")