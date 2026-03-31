import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig,
    PreTrainedConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.constants import OBS_STATE, OBS_IMAGES, ACTION
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from plugrl_server.common.data_utils import torch_tree_batch_size, torch_tree_to_device
from plugrl_server.policy.base_policy_gradient_diffusion_policy import (
    BasePolicyGradientDiffusionPolicyConfig,
    BasePolicyGradientDiffusionPolicy,
    TorchTree,
)
from plugrl_server.policy.registration import register_policy, register_policy_config

UID = "lerobot-diffusion-policy"


@register_policy_config(UID)
@dataclasses.dataclass
class LeRobotDiffusionPolicyConfig(BasePolicyGradientDiffusionPolicyConfig):
    task: str = "pusht"
    pretrained_path: pathlib.Path | None = None
    num_denoising_steps: int = 10
    train_unet_only: bool = True


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(512, 128), output_dim: int = 1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


@register_policy(UID)
class LeRobotDiffusionPolicy(BasePolicyGradientDiffusionPolicy):
    def __init__(self, config: LeRobotDiffusionPolicyConfig):
        super().__init__(config)
        assert config.pretrained_path is not None, "pretrained_path must be specified."
        policy_config = PreTrainedConfig.from_pretrained(config.pretrained_path)
        assert isinstance(policy_config, DiffusionConfig), (
            f"Expected DiffusionConfig, got {type(policy_config)}"
        )
        policy = DiffusionPolicy.from_pretrained(
            config.pretrained_path, config=policy_config
        )
        self.actor: DiffusionPolicy = policy
        self.action_dim = policy_config.output_features["action"].shape[0]
        self.num_denoising_steps = config.num_denoising_steps
        self.action_horizon = policy_config.horizon
        self.critic = ValueHead(
            input_dim=self.actor.diffusion.unet.up_modules[0][0]
            .cond_encoder[1]
            .in_features
            - policy_config.diffusion_step_embed_dim
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = config.task
        if config.train_unet_only:
            self.freeze_conditional_encoder()
        self.to(self.device)

    def _get_timesteps(self) -> torch.Tensor:
        self.actor.diffusion.noise_scheduler.set_timesteps(self.num_denoising_steps)
        return self.actor.diffusion.noise_scheduler.timesteps

    def _initialize_x(self, batch_size: int) -> torch.Tensor:
        x = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device
        )
        return x

    def prepare_observation(self, _obs: dict) -> dict[str, np.ndarray]:
        n_obs_steps = self.actor.config.n_obs_steps

        state_map, image_map = [], []
        if self.task == "pusht":
            state_map = dict(agent_pos="observation.state")
            image_map = dict(pixels="observation.image")
        else:
            raise NotImplementedError(f"Task {self.task} not implemented.")

        obs = dict()
        for i, key in enumerate(state_map):
            obs[state_map[key]] = (
                np.stack(
                    [_obs["states"][f"{key}_{j}"] for j in range(n_obs_steps)], axis=1
                )
                .astype(np.float32)
            )
        for i, key in enumerate(image_map):
            obs[image_map[key]] = (
                np.stack(
                    [_obs["images"][f"{key}_{j}"] for j in range(n_obs_steps)],
                    axis=1,
                )
                .astype(np.float32)
                .transpose(0, 1, 4, 2, 3)
                / 255.0
            )  # B, T, C, H, W
        obs_normalized = self.actor.normalize_inputs(
            dict((key, torch.from_numpy(value).to(self.device)) for key, value in obs.items())
        )
        obs_normalized[OBS_IMAGES] = torch.stack(
            [obs_normalized[key] for key in self.actor.config.image_features], dim=-4
        )
        obs_normalized.pop(*self.actor.config.image_features)
        return dict(
            (key, value.detach().cpu().numpy()) for key, value in obs_normalized.items()
        )

    def fake_diffusion_cond(self, batch_size: int) -> TorchTree:
        n_obs_steps = self.actor.config.n_obs_steps
        n_cams = len(self.actor.config.image_features)
        cond = dict()
        if n_cams > 0:
            cam_name = list(self.actor.config.image_features.keys())[0]
            cam_shape = self.actor.config.image_features[cam_name].shape
            cond[OBS_IMAGES] = torch.zeros(
                (batch_size, n_obs_steps, n_cams, *cam_shape),
            )
        state_dims = self.actor.config.input_features[OBS_STATE].shape[0]
        cond[OBS_STATE] = torch.zeros(
            (batch_size, n_obs_steps, state_dims),
        )
        return cond

    def build_obs_cache(self, obs: TorchTree) -> Any:
        assert isinstance(obs, Mapping), "LeRobot expects mapping-like observations."
        cond_cache = self.actor.diffusion._prepare_global_conditioning(
            torch_tree_to_device(obs, self.device)
        )
        return cond_cache

    def previous_timestep(
        self, noise_scheduler: DDPMScheduler, t: torch.Tensor
    ) -> torch.Tensor:
        if noise_scheduler.custom_timesteps or noise_scheduler.num_inference_steps:
            indices = (noise_scheduler.timesteps == t.unsqueeze(-1)).int().argmax(-1)
            next_indices = indices + 1
            is_last = next_indices >= len(noise_scheduler.timesteps)
            safe_indices = torch.clamp(
                next_indices, max=len(noise_scheduler.timesteps) - 1
            )
            prev_t = noise_scheduler.timesteps[safe_indices]
            prev_t = torch.where(is_last, torch.tensor(-1, device=t.device), prev_t)
        else:
            prev_t = t - 1
        return prev_t

    def _get_variance(
        self,
        noise_scheduler: DDPMScheduler,
        t: torch.Tensor,
        predicted_variance=None,
        variance_type=None,
    ):
        prev_t = self.previous_timestep(noise_scheduler, t.cpu())

        alpha_prod_t = noise_scheduler.alphas_cumprod[t.cpu()][:, None, None]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0,
            noise_scheduler.alphas_cumprod[prev_t.cpu()],
            noise_scheduler.one,
        )[:, None, None]
        alpha_prod_t = alpha_prod_t.to(t.device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(t.device)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = noise_scheduler.config.variance_type

        if variance_type == "fixed_small":
            variance = variance

        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def _denoising_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree,
        x_next: torch.Tensor | None = None,
        *,
        cond_cache: Any = None,
        sampling_noise_level: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        assert t.shape == (b,)
        assert x.shape == (b, self.action_horizon, self.action_dim), (
            f"Expected x shape ({b}, {self.action_horizon}, {self.action_dim}), got {x.shape}"
        )

        device = self.device
        t = t.to(device).long()
        x = x.to(device)

        b_cond = torch_tree_batch_size(cond)
        if cond_cache is None:
            cond = torch_tree_to_device(cond, device)
            cond_cache = self.build_obs_cache(cond)

        global_cond = cond_cache
        if b_cond * self.num_denoising_steps == b:
            global_cond = torch.repeat_interleave(
                cond_cache, self.num_denoising_steps, dim=0
            )

        model_output = self.actor.diffusion.unet(x, t, global_cond=global_cond)

        noise_scheduler, sample = self.actor.diffusion.noise_scheduler, x
        assert isinstance(noise_scheduler, DDPMScheduler)

        prev_t = self.previous_timestep(noise_scheduler, t.cpu())

        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and noise_scheduler.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        alpha_prod_t = noise_scheduler.alphas_cumprod[t.cpu()][:, None, None]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, noise_scheduler.alphas_cumprod[prev_t], noise_scheduler.one
        )[:, None, None]
        alpha_prod_t = alpha_prod_t.to(device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if noise_scheduler.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif noise_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -noise_scheduler.config.clip_sample_range,
                noise_scheduler.config.clip_sample_range,
            )
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        mean = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )
        device = model_output.device
        if noise_scheduler.variance_type == "fixed_small_log":
            variance = self._get_variance(
                noise_scheduler, t, predicted_variance=predicted_variance
            )
        elif noise_scheduler.variance_type == "learned_range":
            variance = self._get_variance(
                noise_scheduler, t, predicted_variance=predicted_variance
            )
        else:
            variance = (
                self._get_variance(
                    noise_scheduler, t, predicted_variance=predicted_variance
                )
                ** 0.5
            )

        logvar = torch.log(variance)
        if sampling_noise_level is not None:
            std = torch.clamp(torch.exp(logvar), min=sampling_noise_level)
        else:
            std = torch.exp(logvar)

        dist = torch.distributions.Normal(mean, std)
        if x_next is None:
            x_next = dist.sample()

        logprob = dist.log_prob(x_next)
        entropy = dist.entropy()

        return x_next, logprob, entropy

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _get_value(self, obs: TorchTree, obs_cache: Any = None) -> torch.Tensor:
        if obs_cache is None:
            obs_cache = self.build_obs_cache(obs)
        global_cond = obs_cache
        value = self.critic(global_cond).squeeze(-1)
        return value

    def _postprocess_action(self, action: torch.Tensor, obs: TorchTree) -> Any:
        start = self.actor.config.n_obs_steps - 1
        end = start + self.actor.config.n_action_steps
        action = action[:, start:end]
        actions = self.actor.unnormalize_outputs({ACTION: action})[ACTION]
        return actions.cpu().numpy()

    def freeze_conditional_encoder(self):
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.actor.diffusion.unet.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    from plugrl_server.cli import main

    main()
