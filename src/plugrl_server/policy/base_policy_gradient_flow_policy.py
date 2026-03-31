import abc
import numpy as np
import torch
from typing import Any
from .base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicy,
    BasePolicyGradientDiffusionPolicyConfig,
    TorchTree,
)


class BasePolicyGradientFlowPolicyConfig(
    BasePolicyGradientDiffusionPolicyConfig
): ...


class BasePolicyGradientFlowPolicy(BasePolicyGradientDiffusionPolicy):
    dt: float

    @abc.abstractmethod
    def _predict_v(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree | None,
        *,
        processed_cond: Any = None,
    ) -> torch.Tensor: ...

    def _denoising_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree,
        x_next: torch.Tensor | None = None,
        *,
        processed_cond: Any = None,
        sampling_noise_level: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        assert t.shape == (b,)

        device = self.device
        x = x.to(device)
        t = t.to(device)
        v = self._predict_v(x, t, cond, processed_cond=processed_cond)

        if sampling_noise_level is None:
            mean, std = x + self.dt * v, torch.zeros_like(x)
            x_next_tensor = mean if x_next is None else x_next
            logprob = torch.zeros_like(x_next_tensor)
            entropy = torch.zeros_like(x_next_tensor)
            return x_next_tensor, logprob, entropy
        else:
            t_expanded = t[:, None, None]
            sigma_t = sampling_noise_level * torch.sqrt(
                t_expanded / (1 - t_expanded).clamp(min=abs(self.dt))
            )
            mean = (
                x
                + (v + sigma_t**2 / (2 * t_expanded) * (x + (1 - t_expanded) * v))
                * self.dt
            )
            std = sigma_t * np.sqrt(abs(self.dt))

        dist = torch.distributions.Normal(mean, std)
        x_next_tensor = mean + std * torch.randn_like(x) if x_next is None else x_next
        entropy = dist.entropy()
        return x_next_tensor, dist.log_prob(x_next_tensor), entropy
