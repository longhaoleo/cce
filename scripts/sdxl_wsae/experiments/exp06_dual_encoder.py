"""实验 6：双编码器解耦验证（语义/风格干预）。"""

from __future__ import annotations

import os

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig
from ..utils import ensure_dir
from .exp54_causal_intervention import run_exp54_causal_intervention


def run_exp06_dual_encoder(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    output_dir: str,
) -> None:
    """
    实验 6 先复用实验 4 的单特征干预能力。

    建议流程：
    - 先运行一次“语义特征”干预；
    - 再运行一次“风格特征”干预；
    - 对比是否出现内容/风格解耦。
    """
    ensure_dir(output_dir)
    subdir = os.path.join(output_dir, "exp06_dual_encoder")
    ensure_dir(subdir)
    print("实验 6 当前为单特征干预入口，请分别传入语义/风格 feature_id 各跑一次。")
    run_exp54_causal_intervention(model_cfg, sae_cfg, run_cfg, int_cfg, subdir)
