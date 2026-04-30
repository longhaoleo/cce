"""当前 SharedSAE 主线的模型包入口。"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .encoding import build_coords_norm, check_expected_hw, ensure_mode, normalize_timestep, sincos_1d, sincos_2d
from .normalization import apply_block_scale, estimate_block_scales
from .sae import (
    ForwardCache,
    LoRAAdapter,
    SharedSAE,
    SpatialBranch,
    TimeBranch,
    _topk_keep,
    build_trainable_param_groups,
    set_stage_trainable,
    unit_norm_decoder_,
    unit_norm_decoder_grad_adjustment_,
)

__all__ = [
    "TrainConfig",
    "SharedSAE",
    "ForwardCache",
    "LoRAAdapter",
    "TimeBranch",
    "SpatialBranch",
    "_topk_keep",
    "unit_norm_decoder_",
    "unit_norm_decoder_grad_adjustment_",
    "build_trainable_param_groups",
    "set_stage_trainable",
    "save_checkpoint",
    "load_checkpoint",
    "build_coords_norm",
    "check_expected_hw",
    "ensure_mode",
    "normalize_timestep",
    "sincos_1d",
    "sincos_2d",
    "estimate_block_scales",
    "apply_block_scale",
]
