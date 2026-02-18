"""实验 5：结构与画幅控制特征（Micro-Conditioning 相关）。"""

from __future__ import annotations

import os

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig
from ..utils import ensure_dir
from .exp54_intervention_suite import run_exp54_causal_intervention


def run_exp05_structure_aspect(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    output_dir: str,
) -> None:
    """
    实验 5 先复用实验 4 的可控注入框架。

    使用方式：
    - 先离线找到与画幅/裁剪相关的特征 id；
    - 在该实验中对对应特征做 injection。
    """
    ensure_dir(output_dir)
    subdir = os.path.join(output_dir, "exp05_structure_aspect")
    ensure_dir(subdir)
    print("实验 5 当前采用实验 4 的注入实现，请确保选择的 feature_id 已与结构/画幅相关。")
    run_exp54_causal_intervention(model_cfg, sae_cfg, run_cfg, int_cfg, subdir)
