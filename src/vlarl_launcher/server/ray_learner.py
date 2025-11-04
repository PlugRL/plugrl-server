import os
import uuid
import ray
import torch
import torch.distributed as dist
import torch.nn.parallel
from loguru import logger
from typing import Any, Dict

from vlarl_launcher.algorithm.base_algorithm import DDPAlgorithm
from vlarl_launcher.common.checkpoint_manager import Checkpoint

@ray.remote(num_gpus=1)
class DDPWorker:
    def __init__(self, algo_refs: list[ray.ObjectRef], rank: int, world_size: int, master_addr: str, master_port: str):
        self.algo: DDPAlgorithm = ray.get(algo_refs[0])
        self.rank = rank
        self.world_size = world_size
        
        self.device = torch.device("cuda", 0)
        print(self.device, self.algo, self.rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend="nccl", 
            rank=self.rank, 
            world_size=self.world_size,
        )
        print(f"DDPWorker Rank {self.rank}: Initialized process group with master {master_addr}:{master_port}")
        self.algo.set_device(self.device) 

        policy_ddp = torch.nn.parallel.DistributedDataParallel(
            self.algo.policy, 
            device_ids=[self.device], 
            output_device=self.device
        )

        self.algo.activate_ddp(policy_ddp)

    def run_learn(self) -> tuple[Checkpoint | None, Dict[str, Any], int]:    
        global_step, train_info = self.algo.learn()
        
        if self.rank == 0:
            checkpoint = self.algo.create_checkpoint()
            return checkpoint, train_info, global_step
        else:
            return None, {}, global_step

    def load_buffer_data(self, global_step: int, meta_info: dict, buffer_data_dict: dict) -> None:
        self.algo.load_server_data(global_step, meta_info, buffer_data_dict)
        
    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()


@ray.remote(num_cpus=1)
class LearnerActor:
    def __init__(self, 
        learner_algo_refs: list[ray.ObjectRef], 
        *,
        ddp_gpus: list[int],
        master_addr: str | None = None, master_port: str | None = None, 
    ):
        super().__init__()
        self.num_ddp_gpus = len(ddp_gpus)
    
        self.master_addr = master_addr or "127.0.0.1"
        self.master_port = master_port or str(29500 + int(uuid.uuid4().int) % 500)
        logger.info(f"LearnerActor master address: {self.master_addr}, port: {self.master_port}")
        self.workers: list[DDPWorker] = [
            DDPWorker.remote(
                learner_algo_refs, i, self.num_ddp_gpus, self.master_addr, self.master_port
            ) for i in ddp_gpus
        ]
        logger.info(f"LearnerActor initialized with {len(ddp_gpus)} DDP Workers.")

    def learn(self, global_step: int, meta_info: dict, buffer_data_dict) -> tuple[Checkpoint, int, dict]:
        logger.info("Starting DDP training orchestration.")

        load_refs = [worker.load_buffer_data.remote(global_step, meta_info, buffer_data_dict) for worker in self.workers]
        ray.get(load_refs)
        
        train_refs = [worker.run_learn.remote() for worker in self.workers]
        
        rank_0_ref = train_refs[0] 
        checkpoint, train_info, global_step = ray.get(rank_0_ref)
        logger.info(f"Training cycle completed. Global Step: {global_step}. Master Policy updated.")
        
        return checkpoint, global_step, train_info
    
    def get_id(self):
        return os.getpid()