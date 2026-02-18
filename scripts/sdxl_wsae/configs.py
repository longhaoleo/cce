"""配置定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_BLOCKS = [
    # "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    # "unet.up_blocks.0.attentions.0",
    # "unet.up_blocks.0.attentions.1",
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
    # exp51：如果你想“可视化指定特征集合”，可以从 exp53 的 csv 读取 feature_id 列表。
    # - feature_csv: 指向 `top_positive_features.csv`（或同格式的 csv）
    # - feature_k:   从 csv 里取前 K 个（<=0 表示全取）
    # 两者留空则走默认的“每步动态 top-k”逻辑。
    exp51_feature_csv: str = ""
    exp51_feature_k: int = 0
    exp51_feature_coeff_scale: float = 1.0
    waterfall_max_features: int = 1024
    waterfall_norm: str = "row"
    waterfall_cmap: str = "magma"


@dataclass
class CausalInterventionConfig:
    """特征干预配置（统一入口是 exp54，exp05/06/07 复用同一套参数）。"""

    block: str = "unet.mid_block.attentions.0"
    feature_top_k: int = 0  # 从 rank_csv 里取前 K 个
    mode: str = "injection"  # injection | ablation
    scale: float = 1.0  # 全局强度系数（会乘到每个特征的 feature_scales 上）
    # 空间约束（可选）：打破“全图对称性”
    # - none: 不加 mask
    # - gaussian_center: 以中心为峰值的 2D 高斯 mask，边缘权重更小
    spatial_mask: str = "none"
    mask_sigma: float = 0.25  # sigma 的相对尺度（乘 min(H,W) 得到像素尺度）
    # 系数来源：
    # - from_x: 在推理时从当前 x 做 SAE.encode 得到 c_i(x)（默认）
    # - from_csv: 从 exp53 导出的“按 step/t 的激活曲线 csv”读取，每个 step 使用预先统计的系数
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
