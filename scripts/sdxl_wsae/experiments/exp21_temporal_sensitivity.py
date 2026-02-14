"""实验 2.1：时间敏感性（早期/晚期注入）验证。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig
from ..utils import ensure_dir
from .exp04_causal_intervention import run_exp04_causal_intervention


@dataclass
class TemporalWindowConfig:
    """时间窗口配置。"""

    early_start: int = 1000
    early_end: int = 800
    late_start: int = 200
    late_end: int = 0


def run_exp21_temporal_sensitivity(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    tw_cfg: TemporalWindowConfig,
    output_dir: str,
) -> None:
    """同一特征分别在早期/晚期注入，对比结果差异。"""
    ensure_dir(output_dir)
    root = os.path.join(output_dir, "exp21_temporal_sensitivity")
    ensure_dir(root)

    early_cfg = CausalInterventionConfig(**vars(int_cfg))
    early_cfg.mode = "injection"
    early_cfg.t_start = int(tw_cfg.early_start)
    early_cfg.t_end = int(tw_cfg.early_end)
    early_cfg.compare_baseline = True
    run_exp04_causal_intervention(model_cfg, sae_cfg, run_cfg, early_cfg, os.path.join(root, "early_injection"))

    late_cfg = CausalInterventionConfig(**vars(int_cfg))
    late_cfg.mode = "injection"
    late_cfg.t_start = int(tw_cfg.late_start)
    late_cfg.t_end = int(tw_cfg.late_end)
    late_cfg.compare_baseline = True
    run_exp04_causal_intervention(model_cfg, sae_cfg, run_cfg, late_cfg, os.path.join(root, "late_injection"))

    print(f"实验 2.1 完成，输出目录: {root}")
