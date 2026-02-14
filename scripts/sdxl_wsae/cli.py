"""统一 CLI：按实验编号运行对应逻辑。"""

from __future__ import annotations

import argparse

from .configs import (
    CausalInterventionConfig,
    ConceptLocateConfig,
    DEFAULT_BLOCKS,
    ModelConfig,
    RunConfig,
    SAEConfig,
    VizConfig,
)
from .experiments.registry import SUPPORTED_EXPERIMENTS, run_experiment
from .experiments.exp07_clip_alignment import ClipEvalConfig
from .experiments.exp21_temporal_sensitivity import TemporalWindowConfig


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数分层：
    - 通用层：模型路径、采样参数、输出路径；
    - 可视化层：exp51 / exp52 参数；
    - 干预层：exp04/05/06/07/21 的 feature intervention 参数；
    - 评估层：exp07 的 CLIP 文本参数；
    - 时间层：exp21 的早晚注入窗口。
    """
    parser = argparse.ArgumentParser(description="SDXL SAE 实验平台")

    parser.add_argument(
        "--experiment",
        type=str,
        default="exp52",
        choices=list(SUPPORTED_EXPERIMENTS.keys()),
        help="实验编号",
    )

    parser.add_argument("--sae_root", type=str, default="~/sdxl-saes")
    parser.add_argument("--model_id", type=str, default="~/datasets/sd-xl/sdxl_diffusers_fp16")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="fp16")

    parser.add_argument("--prompt", type=str, default="a child hold kitchen knife on the table, scary lighting.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./wsae_res_sdxl_output")

    parser.add_argument("--prefer_k", type=int, default=5, help="检查点选择时优先的 k")
    parser.add_argument("--prefer_hidden", type=int, default=5120, help="检查点选择时优先的 hidden")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        help="需要 hook 和可视化的 block，格式如 unet.*.attentions.*",
    )

    # 实验 51/52：可视化参数（分别对应 topk 与 waterfall）
    parser.add_argument("--sae_top_k", type=int, default=10)
    parser.add_argument("--delta_stride", type=int, default=1)
    parser.add_argument("--overlay_alpha", type=float, default=0.75)
    parser.add_argument("--waterfall_max_features", type=int, default=1024)
    parser.add_argument("--waterfall_norm", type=str, default="row", choices=["row", "global", "none"])
    parser.add_argument("--waterfall_cmap", type=str, default="magma")

    # 实验 4/5/6/7/2.1：干预参数
    parser.add_argument("--int_block", type=str, default="unet.mid_block.attentions.0")
    parser.add_argument("--int_feature_id", type=int, default=0)
    parser.add_argument("--int_mode", type=str, default="injection", choices=["injection", "ablation"])
    parser.add_argument("--int_scale", type=float, default=1.0)
    parser.add_argument("--int_t_start", type=int, default=600)
    parser.add_argument("--int_t_end", type=int, default=200)
    parser.add_argument("--int_step_start", type=int, default=-1, help=">=0 时启用 step 范围下界")
    parser.add_argument("--int_step_end", type=int, default=-1, help=">=0 时启用 step 范围上界")
    parser.add_argument("--no_baseline", action="store_true", help="干预实验中不跑 baseline")

    # 实验 7：CLIP 指标参数
    parser.add_argument("--clip_target_text", type=str, default="red")
    parser.add_argument("--clip_preserve_text", type=str, default="car")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")

    # 实验 2.1：早晚窗口
    parser.add_argument("--early_start", type=int, default=1000)
    parser.add_argument("--early_end", type=int, default=800)
    parser.add_argument("--late_start", type=int, default=200)
    parser.add_argument("--late_end", type=int, default=0)

    # 实验 53：概念定位（TARIS）
    parser.add_argument(
        "--loc_block",
        type=str,
        default="unet.mid_block.attentions.0",
        help="exp53 用：用于定位概念的 SAE block",
    )
    parser.add_argument(
        "--concept_name",
        type=str,
        default="",
        help="exp53 用：概念名（用于组织输出目录，例如 red_vs_blue；留空则不加这一层）",
    )
    parser.add_argument(
        "--pos_prompts",
        nargs="+",
        default=[],
        help="exp53 用：正样本 prompts（每条 prompt 用引号包起来）",
    )
    parser.add_argument(
        "--neg_prompts",
        nargs="+",
        default=[],
        help="exp53 用：负样本 prompts（每条 prompt 用引号包起来）",
    )
    parser.add_argument("--taris_t_start", type=int, default=800, help="exp53 用：时间窗口上界（高噪侧）")
    parser.add_argument("--taris_t_end", type=int, default=200, help="exp53 用：时间窗口下界（低噪侧）")
    parser.add_argument("--taris_num_steps", type=int, default=10, help="exp53 用：窗口内采样的步数（<=0 表示全用）")
    parser.add_argument("--taris_delta", type=float, default=1e-6, help="exp53 用：能量归一化分母的稳定项")
    parser.add_argument("--taris_top_k", type=int, default=20, help="exp53 用：输出 top-k 特征数量（正/负各一份）")

    return parser.parse_args()


def build_configs(args: argparse.Namespace):
    """根据 CLI 参数构建实验所需的配置对象。"""
    model_cfg = ModelConfig(
        model_id=args.model_id,
        device=args.device,
        dtype_name=args.dtype,
    )
    sae_cfg = SAEConfig(
        sae_root=args.sae_root,
        blocks=tuple(args.blocks),
        prefer_k=args.prefer_k,
        prefer_hidden=args.prefer_hidden,
    )
    run_cfg = RunConfig(
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    viz_cfg = VizConfig(
        output_dir=args.output_dir,
        sae_top_k=args.sae_top_k,
        delta_stride=args.delta_stride,
        overlay_alpha=args.overlay_alpha,
        waterfall_max_features=args.waterfall_max_features,
        waterfall_norm=args.waterfall_norm,
        waterfall_cmap=args.waterfall_cmap,
    )

    # 约定：<0 表示“禁用该 step 边界”
    step_start = None if int(args.int_step_start) < 0 else int(args.int_step_start)
    step_end = None if int(args.int_step_end) < 0 else int(args.int_step_end)
    int_cfg = CausalInterventionConfig(
        block=args.int_block,
        feature_id=int(args.int_feature_id),
        mode=args.int_mode,
        scale=float(args.int_scale),
        t_start=int(args.int_t_start),
        t_end=int(args.int_t_end),
        step_start=step_start,
        step_end=step_end,
        compare_baseline=not bool(args.no_baseline),
    )
    clip_cfg = ClipEvalConfig(
        target_text=args.clip_target_text,
        preserve_text=args.clip_preserve_text,
        model_name=args.clip_model_name,
    )
    tw_cfg = TemporalWindowConfig(
        early_start=int(args.early_start),
        early_end=int(args.early_end),
        late_start=int(args.late_start),
        late_end=int(args.late_end),
    )

    concept_cfg = ConceptLocateConfig(
        block=str(args.loc_block),
        concept_name=str(args.concept_name),
        pos_prompts=tuple(args.pos_prompts),
        neg_prompts=tuple(args.neg_prompts),
        t_start=int(args.taris_t_start),
        t_end=int(args.taris_t_end),
        num_t_samples=int(args.taris_num_steps),
        delta=float(args.taris_delta),
        top_k=int(args.taris_top_k),
    )

    return model_cfg, sae_cfg, run_cfg, viz_cfg, int_cfg, concept_cfg, clip_cfg, tw_cfg


def main() -> None:
    """程序主入口。"""
    args = parse_args()
    model_cfg, sae_cfg, run_cfg, viz_cfg, int_cfg, concept_cfg, clip_cfg, tw_cfg = build_configs(args)
    run_experiment(
        experiment=args.experiment,
        model_cfg=model_cfg,
        sae_cfg=sae_cfg,
        run_cfg=run_cfg,
        viz_cfg=viz_cfg,
        int_cfg=int_cfg,
        concept_cfg=concept_cfg,
        clip_cfg=clip_cfg,
        tw_cfg=tw_cfg,
    )
