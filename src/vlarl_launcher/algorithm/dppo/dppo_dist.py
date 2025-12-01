import dataclasses
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
import vlarl_launcher.algorithm.dppo.dppo as _dppo
from typing import cast
from vlarl_launcher.algorithm.base_algorithm import DDPAlgorithm
from vlarl_launcher.algorithm.registration import register_algo, register_algo_config
from vlarl_launcher.common.checkpoint_manager import Checkpoint
from vlarl_launcher.common.checkpoint_manager import Checkpoint, move_model_to_cpu, move_optimizer_to_cpu

UID = "dppo-dist"

@register_algo_config(UID)
@dataclasses.dataclass
class DPPOAlgoDistributedConfig(_dppo.DPPOAlgoConfig):
    ...

@register_algo(UID)
class DPPOAlgoDistributed(_dppo.DPPOAlgorithm, DDPAlgorithm):
    policy: _dppo.BasePolicyGradientDiffusionPolicy | torch.nn.parallel.DistributedDataParallel
    
    def __init__(self, config: DPPOAlgoDistributedConfig, policy: _dppo.BasePolicyGradientDiffusionPolicy):
        super().__init__(config, policy)
        self.ddp_enabled = False
        self.checkpoint_cache: Checkpoint | None = None
    
    @property
    def active_policy(self) -> _dppo.BasePolicyGradientDiffusionPolicy:
        return cast(
            _dppo.BasePolicyGradientDiffusionPolicy, 
            self.policy.module if self.ddp_enabled else self.policy
        )
        
    def create_dataloaders(self) -> tuple[Sampler | None, DataLoader]:
        if self.ddp_enabled:
            sampler = torch.utils.data.DistributedSampler(
                self.rollout_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )
            assert self.config.batch_size % dist.get_world_size() == 0, \
                f"Batch size {self.config.batch_size} not divisible by world size {dist.get_world_size()}."
            batch_size = self.config.batch_size // dist.get_world_size()
        else:
            sampler = None
            batch_size = self.config.batch_size
        
        return sampler, torch.utils.data.DataLoader(
            self.rollout_buffer,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=self.rollout_buffer.collate_fn,
        )
        
    def activate_ddp(self, ddp_policy: torch.nn.parallel.DistributedDataParallel) -> None:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.policy = ddp_policy # type: ignore
        self.ddp_enabled = True
        torch.cuda.empty_cache()

    def get_server_data(self) -> tuple[int, dict, dict]:
        return self.global_step, dict(_episode_stats=self._episode_stats), self.rollout_buffer.as_dict()

    def load_server_data(self, global_step: int, meta_info: dict, data: dict) -> None:
        self.global_step = global_step
        self._episode_stats = meta_info["_episode_stats"]
        self.rollout_buffer.load_dict(data)

    def load_learner_state(self, checkpoint: Checkpoint) -> None:
        assert checkpoint.model is not None, "Checkpoint does not contain model state."
        self.policy.load_state_dict(checkpoint.model)
        self.checkpoint_cache = checkpoint
        self.curr_train_itrs = checkpoint.meta.get("train_itrs", 0)
        
    def create_ddp_checkpoint(self) -> Checkpoint:
        self.last_saved_itr = self.curr_train_itrs
        if self.checkpoint_cache is not None:
            model = self.checkpoint_cache.model
            optimizer = self.checkpoint_cache.optimizer
        else:
            assert self.ddp_enabled, "DDP is not enabled. Cannot create DDP checkpoint."
            model = move_model_to_cpu(self.active_policy.state_dict())
            optimizer = move_optimizer_to_cpu({
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            })
        return Checkpoint(
            step=self.global_step,
            model=model,
            optimizer=optimizer,
            meta={
                "train_itrs": self.curr_train_itrs,
                "last_saved_itr": self.last_saved_itr,
            }
        )