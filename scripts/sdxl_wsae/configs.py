"""配置定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_BLOCKS = [
    "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    "unet.up_blocks.0.attentions.0",
    "unet.up_blocks.0.attentions.1",
]


@dataclass
class ModelConfig:
    """SDXL 模型基础配置。"""

    sdxl_unbox_root: str = "~/sdxl-unbox"
    model_id: str = "~/datasets/sd-xl/sdxl_diffusers_fp16"
    device: str = "cuda"
    dtype_name: str = "fp16"


@dataclass
class SAEConfig:
    """SAE 相关配置。"""

    sae_root: str = "~/sdxl-saes"
    blocks: Tuple[str, ...] = tuple(DEFAULT_BLOCKS)
    prefer_k: int = 10
    prefer_hidden: int = 5120


@dataclass
class RunConfig:
    """采样运行配置。"""

    prompt: str
    steps: int = 25
    guidance_scale: float = 8.0
    seed: int = 42


@dataclass
class VizConfig:
    """可视化配置。"""

    output_dir: str = "./wsae_res_sdxl_output"
    sae_top_k: int = 10
    delta_stride: int = 1
    overlay_alpha: float = 0.75
    module: str = "topk"
    waterfall_max_features: int = 1024
    waterfall_norm: str = "row"
    waterfall_cmap: str = "magma"


@dataclass
class CausalInterventionConfig:
    """实验 4/2.x 使用的特征干预配置。"""

    block: str = "unet.mid_block.attentions.0"
    feature_id: int = 0
    mode: str = "injection"  # injection | ablation
    scale: float = 1.0
    t_start: int = 600
    t_end: int = 200
    step_start: Optional[int] = None
    step_end: Optional[int] = None
    compare_baseline: bool = True


@dataclass
class ConceptLocateConfig:
    """实验 53：概念定位（TARIS）配置。"""

    block: str = "unet.mid_block.attentions.0"
    concept_name: str = ""
    pos_prompts: Tuple[str, ...] = ()
    neg_prompts: Tuple[str, ...] = ()
    t_start: int = 800
    t_end: int = 200
    num_t_samples: int = 10
    delta: float = 1e-6
    top_k: int = 20
