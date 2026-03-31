from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import torch

from plugrl_server.buffer.rollout_buffer import GAEBuffer
from plugrl_server.common.data_utils import (
    numpy_tree_to_torch,
    stack_numpy_tree,
    torch_tree_to_device,
)
from plugrl_server.policy.base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
)
from plugrl_server.policy.state import PolicyTrainState

from .utils import compute_cfm_loss

TorchTree: TypeAlias = torch.Tensor | dict[str, "TorchTree"]


class FPOBuffer(GAEBuffer):
    loss_eps: np.ndarray
    loss_t: np.ndarray
    initial_cfm_loss: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        example_train_state: PolicyTrainState,
        *,
        n_samples_per_action: int,
        discretize_t_for_training: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__(
            buffer_size=buffer_size,
            example_train_state=example_train_state,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.n_samples_per_action = n_samples_per_action
        self.discretize_t_for_training = discretize_t_for_training
        action_shape = self.actions.shape[1:]
        self.loss_eps = np.empty(
            (buffer_size, n_samples_per_action) + action_shape,
            dtype=self.actions.dtype,
        )
        self.loss_t = np.empty(
            (buffer_size, n_samples_per_action, 1),
            dtype=np.float32,
        )
        self.initial_cfm_loss = np.empty(
            (buffer_size, n_samples_per_action),
            dtype=np.float32,
        )

    def reset(self):
        super().reset()
        self.loss_eps.fill(0)
        self.loss_t.fill(0)
        self.initial_cfm_loss.fill(0)

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= self.idx:
            raise IndexError("FPOBuffer index out of range")
        return (
            self.train_state_storage.get_item(idx),
            self.actions[idx],
            self.values[idx],
            self.advantages[idx],
            self.returns[idx],
            self.loss_eps[idx],
            self.loss_t[idx],
            self.initial_cfm_loss[idx],
        )

    def collate_fn(self, batch: list[tuple]) -> tuple:
        obs_items, *rest = zip(*batch)
        return (
            torch_tree_to_device(
                numpy_tree_to_torch(stack_numpy_tree(list(obs_items), axis=0)),
                torch.device("cpu"),
            ),
            *(torch.from_numpy(np.stack(items, axis=0)) for items in rest),
        )

    def prepare_fpo_fields(
        self, policy: BasePolicyGradientFlowPolicy, *, batch_size: int
    ) -> None:
        if self.idx == 0:
            return
        loss_eps = self._sample_loss_eps(policy)
        loss_t = self._sample_loss_t(policy, batch_size=self.idx)

        self.loss_eps[: self.idx] = loss_eps.detach().cpu().numpy()
        self.loss_t[: self.idx] = loss_t.detach().cpu().numpy()
        for i in range(0, self.idx, batch_size):
            j = min(i + batch_size, self.idx)
            obs = self.train_state_storage.get_item(slice(i, j))
            obs_torch = torch_tree_to_device(numpy_tree_to_torch(obs), policy.device)
            obs_cache = policy.build_obs_cache(obs_torch)
            actions = torch.from_numpy(self.actions[i:j]).to(policy.device)
            initial_cfm_loss = compute_cfm_loss(
                policy,
                obs_torch,
                actions,
                loss_eps=loss_eps[i:j],
                loss_t=loss_t[i:j],
                obs_cache=obs_cache,
            )
            self.initial_cfm_loss[i:j] = initial_cfm_loss.detach().cpu().numpy()

    def _sample_loss_eps(self, policy: BasePolicyGradientFlowPolicy) -> torch.Tensor:
        samples = policy._initialize_x(self.idx * self.n_samples_per_action)
        return samples.reshape(self.idx, self.n_samples_per_action, *self.actions.shape[1:])

    def _sample_loss_t(
        self, policy: BasePolicyGradientFlowPolicy, *, batch_size: int
    ) -> torch.Tensor:
        if self.discretize_t_for_training:
            timesteps = policy._get_timesteps().to(policy.device)
            indices = torch.randint(
                0,
                timesteps.shape[0],
                (batch_size, self.n_samples_per_action),
                device=policy.device,
            )
            return timesteps[indices].unsqueeze(-1)
        return torch.rand(
            (batch_size, self.n_samples_per_action, 1),
            device=policy.device,
        )
