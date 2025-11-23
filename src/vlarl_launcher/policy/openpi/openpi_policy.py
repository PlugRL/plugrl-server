import dataclasses

from openpi.policies import policy_config as _policy_config
from openpi import transforms as _transforms
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models import gemma as _gemma

from tensordict import TensorDict
from tensordict._pytree import TensorDict
from tensordict import tensordict
import torch
import pathlib
import numpy as np
import jax
from typing import Any

from vlarl_launcher.common.data_utils import batch_aggregate, unbatch_aggregate
from .openpi_transforming import get_transform
from ..base_policy_gradient_diffusion_policy import BasePolicyGradientDiffusionPolicy, BasePolicyGradientDiffusionPolicyConfig
from ..registration import register_policy, register_policy_config
from .value_head import ValueHead

UID = "pi0-policy"

@register_policy_config(UID, supported_algos=[("dppo", "libero")])
@dataclasses.dataclass
class Pi0PolicyConfig(BasePolicyGradientDiffusionPolicyConfig):
    name: str = "pi05_tiny_libero"
    checkpoint_path: pathlib.Path | None = None
    default_prompt: str | None = None
    denoising_steps: int = 10
    train_expert_only: bool = True
    
@register_policy(UID)
class Pi0Policy(BasePolicyGradientDiffusionPolicy):
    actor: _model.pi0_pytorch.PI0Pytorch
    config: Pi0PolicyConfig

    def __init__(self, config: Pi0PolicyConfig):
        super().__init__(config)
        train_config = _config.get_config(config.name)
        checkpoint_path = config.checkpoint_path
        assert checkpoint_path is not None, "checkpoint_path must be provided in the config"
        weight_path = checkpoint_path / "model.safetensors"
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(
            str(checkpoint_path / "assets"), data_config.asset_id
        )
        pytorch_device = config.device
        repack_transforms = get_transform(config.name)

        self.actor = model
        self.input_transform = _transforms.compose([
            *repack_transforms,
            _transforms.InjectDefaultPrompt(config.default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])
        self.output_transform = _transforms.compose([
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ])
        self.actor = self.actor.to(pytorch_device)
        self.max_token_len = train_config.model.max_token_len
        self.action_dim = train_config.model.action_dim
        self.action_horizon = train_config.model.action_horizon
        self.num_denoising_steps = config.denoising_steps
        self.dt = -1.0 / self.num_denoising_steps
        self.actor.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        self.actor.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        paligemma_config = _gemma.get_config(train_config.model.paligemma_variant)
        self.critic = ValueHead(
            input_dim=paligemma_config.width,
            hidden_sizes=(512, 256, 128),
            output_dim=1,
            activation="relu",
            bias_last=True,
        ).to(pytorch_device, dtype=torch.bfloat16)
        
        if config.train_expert_only:
            self.freeze_vlm()

    def _get_timesteps(self) -> torch.Tensor:
        timestep = torch.linspace(1, 1. / self.num_denoising_steps, self.num_denoising_steps)
        return timestep

    def _initialize_x(self, obs: TensorDict) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        return x
    
    def prepare_observation(self, _obs: dict) -> torch.Tensor | TensorDict:
        unbatch_obs = unbatch_aggregate(_obs)
        batch_size = len(unbatch_obs)
        obs_list = []
        for single_obs in unbatch_obs:
            inputs = jax.tree.map(lambda x: x, single_obs)
            inputs = self.input_transform(inputs)
            inputs = jax.tree.map(lambda x: x[np.newaxis, ...], inputs)
            obs_list.append(inputs)
            
        batch_obs = batch_aggregate(obs_list)
        batch_obs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)), batch_obs)
        return TensorDict(batch_obs, batch_size=[batch_size])

    def preprocess_observation(self, obs: TensorDict) -> Any:
        obs_dict = obs.to(self.device).to_dict()
        observation = _model.Observation.from_dict(obs_dict)
        images, img_masks, lang_tokens, lang_masks, state = self.actor._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.actor.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        prefix_att_2d_masks_4d = self.actor._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        outputs, past_key_values = self.actor.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return state, prefix_pad_masks, outputs, past_key_values

    def fake_diffusion_cond(self, batch_size: int) -> tensordict.TensorDict:
        return tensordict.TensorDict(
            dict(
                image=dict(
                    base_0_rgb=torch.zeros(batch_size, 224, 224, 3, dtype=torch.uint8),
                    left_wrist_0_rgb=torch.zeros(batch_size, 224, 224, 3, dtype=torch.uint8),
                    right_wrist_0_rgb=torch.zeros(batch_size, 224, 224, 3, dtype=torch.uint8),
                ),
                image_mask=dict(
                    base_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
                    left_wrist_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
                    right_wrist_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
                ),
                state=torch.zeros(batch_size, self.action_dim),
                tokenized_prompt=torch.zeros(batch_size, self.max_token_len, dtype=torch.long),
                tokenized_prompt_mask=torch.zeros(batch_size, self.max_token_len, dtype=torch.bool),
            ),
            batch_size=[batch_size]
        )
        
    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(self, action: torch.Tensor, obs: tensordict.TensorDict) -> Any:
        outputs = {
            "state": obs.get("state"),
            "actions": action
        }
        outputs = jax.tree.map(lambda x: x.detach().cpu().numpy(), outputs)
        unbatched_outputs = unbatch_aggregate(outputs)
        actions = []
        for out in unbatched_outputs:
            transformed_out = self.output_transform(out)
            actions.append(transformed_out['actions'])
        return np.stack(actions, axis=0)
    
    def _denoising_step(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: tensordict.TensorDict, 
        x_next: torch.Tensor | None = None,
        *,
        processed_cond: Any = None,
        sampling_noise_level: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        assert t.shape == (b,)
        assert x.shape == (b, self.action_horizon, self.action_dim)
        
        device = self.device
        t = t.to(device)
        x = x.to(device)
        
        b_cond = cond.shape[0]        
        if processed_cond is None:
            cond = cond.to(device)
            processed_cond = self.preprocess_observation(cond)
        state, prefix_pad_masks, outputs, past_key_values = processed_cond
        
        if b_cond * self.num_denoising_steps == b:
            state = torch.repeat_interleave(state, self.num_denoising_steps, dim=0)
            prefix_pad_masks = torch.repeat_interleave(prefix_pad_masks, self.num_denoising_steps, dim=0)
            past_key_values.batch_repeat_interleave(self.num_denoising_steps)
        
        vt = self.actor.denoise_step(
            state, prefix_pad_masks, past_key_values, x, t
        )
        
        if sampling_noise_level is None:
            mean, std = x + self.dt * vt, torch.zeros_like(x)
        else:
            t_expanded = t[:, None, None]
            sigma_t = sampling_noise_level * torch.sqrt(t_expanded / (1 - t_expanded).clamp(min=abs(self.dt)))
            mean = x + (
                vt + sigma_t**2 / (2 * t_expanded) * (x + (1 - t_expanded) * vt)
            ) * self.dt
            std = sigma_t * np.sqrt(abs(self.dt))
            
        dist = torch.distributions.Normal(mean, std)

        if x_next is None:
            noise = torch.randn_like(x)
            x_next = mean + std * noise

        if sampling_noise_level is not None:
            logprob = dist.log_prob(x_next)
        else:
            logprob = torch.zeros_like(x_next)
        entropy = dist.entropy()
        
        return x_next, logprob, entropy

        
    def _get_value(self, obs: TensorDict, processed_obs: Any = None) -> torch.Tensor:
        b = obs.shape[0]
        if processed_obs is None:
            processed_obs = self.preprocess_observation(obs)
        _, _, outputs, _ = processed_obs
        hidden_state = outputs[0][:, -1]
        value = self.critic(hidden_state).squeeze(-1)
        return value
    
    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.actor.paligemma_with_expert.paligemma.eval()
            for params in self.actor.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False