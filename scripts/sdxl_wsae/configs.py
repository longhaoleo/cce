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
    # exp51：fixed 模式下按 concept 自动读取 exp53 的 top_positive_features.csv。
    exp51_mode: str = "dynamic"  # dynamic | fixed
    exp51_feature_k: int = 0
    exp51_feature_coeff_scale: float = 1.0
    exp51_concept: str = ""
    waterfall_max_features: int = 1024
    waterfall_norm: str = "row"
    waterfall_cmap: str = "magma"


@dataclass
class CausalInterventionConfig:
    """特征干预配置（exp54 及复用入口）。"""

    blocks: Tuple[str, ...] = ("unet.up_blocks.0.attentions.0",)
    targetconcept: str = "concept"
    feature_top_k: int = 0  # 从 rank_csv 里取前 K 个
    mode: str = "injection"  # injection | ablation
    scale: float = 1.0  # 全局强度系数（会乘到每个特征的 feature_scales 上）
    use_time_weight: bool = True  # True: 在 from_x 的 c_i(x) 上乘 exp53 的按 step 时间权重；False: 仅 from_x
    use_spatial_norm_weight: bool = False  # True: 乘空间范数归一化权重
    # 空间约束（可选）：打破“全图对称性”
    # - none: 不加 mask
    # - gaussian_center: 以中心为峰值的 2D 高斯 mask，边缘权重更小
    spatial_mask: str = "none"
    mask_sigma: float = 0.25  # sigma 的相对尺度（乘 min(H,W) 得到像素尺度）
    # 时间权重 csv（exp53 导出的按 step 曲线），用于与 c_i(x) 相乘
    coeff_csv: str = ""  # out_concept_dict/<concept>/feature_time_scores.csv
    t_start: int = 600
    t_end: int = 200
    step_start: Optional[int] = None
    step_end: Optional[int] = None
    compare_baseline: bool = True  # 是否输出 baseline（不干预）对照


@dataclass
class ConceptLocateConfig:
    """实验 53：概念定位（TARIS）配置。"""

    block: str = "unet.mid_block.attentions.0"
    concept_name: str = ""
    t_start: int = 800
    t_end: int = 200
    num_t_samples: int = 10
    delta: float = 1e-6
    top_k: int = 20


@dataclass
class TemporalWindowConfig:
    """早期/晚期时间窗口（用于蝴蝶效应/时间敏感性对比）。"""

    early_start: int = 1000
    early_end: int = 800
    late_start: int = 200
    late_end: int = 0
