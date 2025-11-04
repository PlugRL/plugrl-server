import os
import ray
from vlarl_launcher.policy.registration import REGISTERED_POLICY_CONFIGS, make_policy

policy = make_policy("dummy-policy", config=REGISTERED_POLICY_CONFIGS["dummy-policy"])

@ray.remote(num_cpus=1)
class LearnerActor:
    def __init__(self, 
        # learner_algo_ref: ray.ObjectRef, 
        # ddp_gpus: list[int],
        # master_addr: str | None = None, master_port: str | None = None, 
    ):
        super().__init__()
        
    def get_id(self):
        return os.getpid()
    
ray.init()
print("Ray initialized.")
print(f"Available resources: {ray.available_resources()}")
actor = LearnerActor.remote()
actor_id = ray.get(actor.get_id.remote())
print(f"Actor is running in process ID: {actor_id}")