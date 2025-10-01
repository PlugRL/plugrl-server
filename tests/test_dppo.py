import math
import hydra
from dppo.model.diffusion.diffusion import DiffusionModel
from omegaconf import OmegaConf
import torch
import pathlib
import numpy as np
from vlarl_launcher import PACKAGE_DIR

cfg = OmegaConf.load(PACKAGE_DIR / "meta" / "dppo" / "cfg" / "robomimic" / "square.yaml")
OmegaConf.resolve(cfg)

model: DiffusionModel = hydra.utils.instantiate(cfg.model)

checkpoint_path = pathlib.Path("pretrains/dppo/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])

normalization_path = PACKAGE_DIR / "meta" / "dppo" / "asset" / "robomimic" / "square" / "normalization.npz"
normalization = np.load(normalization_path)
print(normalization)