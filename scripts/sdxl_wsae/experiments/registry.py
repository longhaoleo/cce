"""实验注册与分发。"""

from __future__ import annotations

from typing import Callable, Dict

from ..configs import (
    CausalInterventionConfig,
    ConceptLocateConfig,
    RunConfig,
    SAEConfig,
    TemporalWindowConfig,
    VizConfig,
)
from .exp01_red_car_surgery import run_exp01_red_car_surgery
from .exp02_context_preservation import run_exp02_context_preservation
from .exp51_feature_dynamics_topk import run_exp51_feature_dynamics_topk
from .exp52_feature_dynamics_waterfall import run_exp52_feature_dynamics_waterfall
from .exp05_structure_aspect import run_exp05_structure_aspect
from .exp06_dual_encoder import run_exp06_dual_encoder
from .exp07_clip_alignment import ClipEvalConfig, run_exp07_clip_alignment
from .exp54_intervention_suite import run_exp54_intervention_suite
from .exp22_feature_inertia import run_exp22_feature_inertia
from .exp23_semantic_arithmetic import run_exp23_semantic_arithmetic
from .exp24_orthogonality import run_exp24_orthogonality
from .exp25_trajectory_grafting import run_exp25_trajectory_grafting
from .exp53_concept_locator_taris import run_exp53_concept_locator_taris


SUPPORTED_EXPERIMENTS = {
    "exp01": "组合概念外科手术（Red Car）",
    "exp02": "背景与主体无损分离",
    "exp51": "特征动力学 Top-K 热图叠加",
    "exp52": "特征动力学瀑布图（Money Plot）",
    "exp53": "概念定位（TARIS 时域平均相对重要性）",
    "exp05": "结构与画幅控制特征",
    "exp06": "双编码器解耦验证",
    "exp07": "CLIP 对齐定量评估",
    "exp54": "蝴蝶效应测试（Early vs Late Injection）",
    "exp22": "特征惯性（单步/多步擦除）",
    "exp23": "语义算术与特征交换",
    "exp24": "特征正交性检验",
    "exp25": "跨轨迹嫁接",
}


def run_experiment(
    *,
    experiment: str,
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    viz_cfg: VizConfig,
    int_cfg: CausalInterventionConfig,
    concept_cfg: ConceptLocateConfig,
    clip_cfg: ClipEvalConfig,
    tw_cfg: TemporalWindowConfig,
) -> None:
    """按实验编号分发到对应实现。"""
    exp = str(experiment).lower()
    # 统一分发表：便于后续新增实验时只改一处。
    dispatch: Dict[str, Callable[[], None]] = {
        "exp01": lambda: run_exp01_red_car_surgery(viz_cfg.output_dir),
        "exp02": lambda: run_exp02_context_preservation(viz_cfg.output_dir),
        
        "exp51": lambda: run_exp51_feature_dynamics_topk(model_cfg, sae_cfg, run_cfg, viz_cfg),
        "exp52": lambda: run_exp52_feature_dynamics_waterfall(model_cfg, sae_cfg, run_cfg, viz_cfg),
        "exp53": lambda: run_exp53_concept_locator_taris(model_cfg, sae_cfg, run_cfg, concept_cfg, viz_cfg.output_dir),
        "exp54": lambda: run_exp54_intervention_suite(model_cfg, sae_cfg, run_cfg, int_cfg, tw_cfg, viz_cfg.output_dir),

        "exp05": lambda: run_exp05_structure_aspect(model_cfg, sae_cfg, run_cfg, int_cfg, viz_cfg.output_dir),
        "exp06": lambda: run_exp06_dual_encoder(model_cfg, sae_cfg, run_cfg, int_cfg, viz_cfg.output_dir),
        "exp07": lambda: run_exp07_clip_alignment(model_cfg, sae_cfg, run_cfg, int_cfg, clip_cfg, viz_cfg.output_dir),
        "exp22": lambda: run_exp22_feature_inertia(viz_cfg.output_dir),
        "exp23": lambda: run_exp23_semantic_arithmetic(viz_cfg.output_dir),
        "exp24": lambda: run_exp24_orthogonality(viz_cfg.output_dir),
        "exp25": lambda: run_exp25_trajectory_grafting(viz_cfg.output_dir),
    }

    fn = dispatch.get(exp)
    if fn is None:
        raise ValueError(f"不支持的实验编号: {experiment}，可选: {', '.join(SUPPORTED_EXPERIMENTS.keys())}")
    fn()
