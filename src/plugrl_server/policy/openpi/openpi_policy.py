import dataclasses
from collections.abc import Mapping
from typing import Any

from openpi import transforms as _transforms
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models import gemma as _gemma
from openpi.policies import libero_policy as _libero_policy

import pathlib
import jax
import numpy as np
import torch

from plugrl_server.common.data_utils import (
    batch_aggregate,
    torch_tree_to_device,
    torch_tree_batch_size,
    unbatch_aggregate,
)
from plugrl_server.common.logging_utils import get_logger
from .openpi_transforming import get_transform
from .debug_artifacts import write_debug_artifacts
from .robocasa_norm_stats import load_robocasa_norm_stats
from ..base_policy_gradient_diffusion_policy import TorchTree
from ..base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
    BasePolicyGradientFlowPolicyConfig,
)
from ..registration import register_policy, register_policy_config
from .value_head import ValueHead

UID = "pi0-policy"
logger = get_logger(__name__)


def _is_robocasa_config(config_name: str) -> bool:
    return "robocasa" in config_name.lower()


def _describe_transform_sequence(
    transforms: list[_transforms.DataTransformFn],
) -> list[str]:
    return [type(transform).__name__ for transform in transforms]


def _make_example_observation(config_name: str) -> dict[str, Any] | None:
    config_name_lower = config_name.lower()
    if "robocasa" in config_name_lower:
        return {
            "images": {
                "robot0_agentview_left": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "robot0_agentview_right": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
                "robot0_eye_in_hand": np.random.randint(
                    256, size=(224, 224, 3), dtype=np.uint8
                ),
            },
            "states": {"state": np.random.rand(16).astype(np.float32)},
            "text": "close the blender lid",
        }
    if "libero" in config_name_lower:
        return _libero_policy.make_libero_example()
    return None


def _repeat_fake_template(template: Any, batch_size: int) -> Any:
    if isinstance(template, np.ndarray):
        repeated = np.broadcast_to(template, (batch_size,) + template.shape).copy()
        return torch.from_numpy(repeated)
    if isinstance(template, Mapping):
        return {
            key: _repeat_fake_template(value, batch_size)
            for key, value in template.items()
        }
    raise TypeError(f"Unsupported fake OpenPI template leaf: {type(template)!r}")


def _to_numpy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        try:
            return np.asarray(value)
        except Exception:
            return [_to_numpy_tree(item) for item in value]
    return np.asarray(value)


@register_policy_config(UID)
@dataclasses.dataclass
class Pi0PolicyConfig(BasePolicyGradientFlowPolicyConfig):
    name: str = "pi05_tiny_libero"
    checkpoint_path: pathlib.Path | None = None
    dataset_dir: pathlib.Path | None = None
    norm_stats_path: pathlib.Path | None = None
    default_prompt: str | None = None
    denoising_steps: int = 5
    train_expert_only: bool = True
    debug_artifact_dir: pathlib.Path | None = None
    export_debug_artifacts: bool = False
    debug_artifact_once: bool = True


@register_policy_config(UID, "robocasa")
@dataclasses.dataclass
class Pi0PolicyConfigRobocasa(Pi0PolicyConfig):
    name: str = "pi05_robocasa_test"


@register_policy(UID)
class Pi0Policy(BasePolicyGradientFlowPolicy):
    actor: _model.pi0_pytorch.PI0Pytorch
    config: Pi0PolicyConfig

    def __init__(self, config: Pi0PolicyConfig):
        super().__init__(config)
        logger.info(
            "Initializing OpenPI policy: config_name=%s checkpoint_path=%s device=%s",
            config.name,
            config.checkpoint_path,
            self.device,
        )
        train_config = _config.get_config(config.name)
        checkpoint_path = config.checkpoint_path
        assert checkpoint_path is not None, (
            "checkpoint_path must be provided in the config"
        )
        weight_path = checkpoint_path / "model.safetensors"
        logger.info("Loading OpenPI model weights from %s", weight_path)
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )
        if _is_robocasa_config(config.name):
            norm_stats, norm_stats_source = load_robocasa_norm_stats(
                train_config,
                dataset_dir=config.dataset_dir,
                norm_stats_path=config.norm_stats_path,
            )
        else:
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load norm stats.")
            norm_stats_source = checkpoint_path / "assets"
            logger.info(
                "Loading OpenPI norm stats from checkpoint assets: %s (asset_id=%s)",
                norm_stats_source,
                data_config.asset_id,
            )
            norm_stats = _checkpoints.load_norm_stats(
                str(norm_stats_source), data_config.asset_id
            )
        pytorch_device = config.device
        repack_transforms = get_transform(config.name)
        input_transforms = [
            *repack_transforms,
            _transforms.InjectDefaultPrompt(config.default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                std_floor=data_config.std_floor,
            ),
            *data_config.model_transforms.inputs,
        ]
        output_transforms = [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                std_floor=data_config.std_floor,
            ),
            *data_config.data_transforms.outputs,
        ]
        logger.info("OpenPI norm stats source: %s", norm_stats_source)
        logger.info(
            "OpenPI input transform pipeline: %s",
            _describe_transform_sequence(input_transforms),
        )
        logger.info(
            "OpenPI output transform pipeline: %s",
            _describe_transform_sequence(output_transforms),
        )

        self.actor = model
        self.input_transform = _transforms.compose(input_transforms)
        self.output_transform = _transforms.compose(output_transforms)
        self.actor = self.actor.to(pytorch_device)
        self.max_token_len = train_config.model.max_token_len
        self.action_dim = train_config.model.action_dim
        self.action_horizon = train_config.model.action_horizon
        self.num_denoising_steps = config.denoising_steps
        self.dt = -1.0 / self.num_denoising_steps
        self.actor.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self._debug_artifact_dir = (
            config.debug_artifact_dir.resolve()
            if config.debug_artifact_dir is not None
            else None
        )
        self._debug_artifacts_written = False
        self._debug_artifact_index = 0

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

        self._fake_cond_template = self._build_fake_cond_template(config.name)

    def _get_timesteps(self) -> torch.Tensor:
        timestep = torch.linspace(
            1, 1.0 / self.num_denoising_steps, self.num_denoising_steps
        )
        return timestep

    def _initialize_x(self, batch_size: int) -> torch.Tensor:
        x = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device
        )
        return x

    def prepare_observation(self, _obs: dict) -> dict[str, Any]:
        unbatch_obs = unbatch_aggregate(_obs)
        obs_list = []
        for single_obs in unbatch_obs:
            raw_inputs = jax.tree.map(lambda x: x, single_obs)
            transformed_inputs = self.input_transform(
                jax.tree.map(lambda x: x, raw_inputs)
            )
            self._maybe_write_debug_artifacts(raw_inputs, transformed_inputs)
            obs_list.append(transformed_inputs)

        batch_obs = batch_aggregate(obs_list)
        return _to_numpy_tree(batch_obs)

    def _build_fake_cond_template(self, config_name: str) -> Any:
        example_obs = _make_example_observation(config_name)
        if example_obs is None:
            logger.warning(
                "No fake OpenPI observation template is registered for config %s. "
                "Falling back to legacy zero templates.",
                config_name,
            )
            return None

        transformed_example = self.input_transform(jax.tree.map(lambda x: x, example_obs))
        logger.info(
            "Built OpenPI fake condition template for %s with keys=%s",
            config_name,
            sorted(transformed_example.keys()),
        )
        return jax.tree.map(lambda x: np.asarray(x), transformed_example)

    def _maybe_write_debug_artifacts(
        self,
        raw_inputs: dict[str, Any],
        transformed_inputs: dict[str, Any],
    ) -> None:
        if not self.config.export_debug_artifacts or self._debug_artifact_dir is None:
            return
        if self.config.debug_artifact_once and self._debug_artifacts_written:
            return

        artifact_dir = (
            self._debug_artifact_dir / "first_observation"
            if self.config.debug_artifact_once
            else self._debug_artifact_dir
            / f"observation_{self._debug_artifact_index:04d}"
        )
        artifact_path = write_debug_artifacts(
            artifact_dir,
            raw_obs=raw_inputs,
            transformed_obs=transformed_inputs,
        )
        self._debug_artifacts_written = True
        self._debug_artifact_index += 1
        logger.info("Saved OpenPI debug artifacts to %s", artifact_path)

    def build_obs_cache(self, obs: TorchTree) -> Any:
        obs_dict = torch_tree_to_device(obs, self.device)
        observation = _model.Observation.from_dict(obs_dict)
        images, img_masks, lang_tokens, lang_masks, state = (
            self.actor._preprocess_observation(observation, train=False)
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.actor.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.actor._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )

        outputs, past_key_values = self.actor.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return state, prefix_pad_masks, outputs, past_key_values

    def fake_diffusion_cond(self, batch_size: int) -> TorchTree:
        if self._fake_cond_template is not None:
            return _repeat_fake_template(self._fake_cond_template, batch_size)
        return dict(
            image=dict(
                base_0_rgb=torch.zeros(batch_size, 224, 224, 3, dtype=torch.uint8),
                left_wrist_0_rgb=torch.zeros(
                    batch_size, 224, 224, 3, dtype=torch.uint8
                ),
                right_wrist_0_rgb=torch.zeros(
                    batch_size, 224, 224, 3, dtype=torch.uint8
                ),
            ),
            image_mask=dict(
                base_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
                left_wrist_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
                right_wrist_0_rgb=torch.zeros(batch_size, dtype=torch.bool),
            ),
            state=torch.zeros(batch_size, self.action_dim),
            tokenized_prompt=torch.zeros(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.zeros(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
        )

    def _iterative_process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _postprocess_action(self, action: torch.Tensor, obs: TorchTree) -> Any:
        assert isinstance(obs, Mapping), "OpenPI expects mapping-like observations."
        outputs = dict(state=obs["state"], actions=action)
        outputs = jax.tree.map(lambda x: x.detach().cpu().numpy(), outputs)
        unbatched_outputs = unbatch_aggregate(outputs)
        actions = []
        for out in unbatched_outputs:
            transformed_out = self.output_transform(out)
            actions.append(transformed_out["actions"])
        return np.stack(actions, axis=0)

    def _predict_v(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TorchTree | None,
        *,
        cond_cache: Any = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        assert t.shape == (b,)
        assert x.shape == (b, self.action_horizon, self.action_dim)

        device = self.device
        t = t.to(device)
        x = x.to(device)

        if cond_cache is None:
            assert cond is not None
            b_cond = torch_tree_batch_size(cond)
            cond = torch_tree_to_device(cond, device)
            cond_cache = self.build_obs_cache(cond)
        else:
            state, _, _, _ = cond_cache
            b_cond = state.shape[0]
        state, prefix_pad_masks, outputs, past_key_values = cond_cache

        if b_cond != b:
            if b % b_cond != 0:
                raise ValueError(
                    f"Cannot repeat OpenPI cond batch {b_cond} to match x batch {b}."
                )
            repeat_factor = b // b_cond
            state = torch.repeat_interleave(state, repeat_factor, dim=0)
            prefix_pad_masks = torch.repeat_interleave(
                prefix_pad_masks, repeat_factor, dim=0
            )
            past_key_values.batch_repeat_interleave(repeat_factor)

        return self.actor.denoise_step(state, prefix_pad_masks, past_key_values, x, t)

    def _get_value(self, obs: TorchTree, obs_cache: Any = None) -> torch.Tensor:
        if obs_cache is None:
            obs_cache = self.build_obs_cache(obs)
        _, _, outputs, _ = obs_cache
        mean_hidden_state = outputs[0].mean(dim=1)
        value = self.critic(mean_hidden_state).squeeze(-1)
        return value

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.actor.paligemma_with_expert.paligemma.eval()
            for params in self.actor.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
