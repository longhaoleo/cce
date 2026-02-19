"""统一 CLI：按实验编号运行对应逻辑。"""

from __future__ import annotations

import argparse
import os
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
from .configs import TemporalWindowConfig


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数分层：
    - 通用层：模型路径、采样参数、输出路径；
    - exp51：top-k 可视化参数；
    - exp52：瀑布图参数；
    - exp53：概念定位（TARIS）参数；
    - exp54：干预参数 + 时间窗参数；
    - exp07：CLIP 评估参数。
    """
    parser = argparse.ArgumentParser(description="SDXL SAE 实验平台（统一入口）")

    g_main = parser.add_argument_group("主参数（通用）")
    g_model = parser.add_argument_group("模型参数（SDXL）")
    g_sae = parser.add_argument_group("SAE 参数（通用）")
    g_exp51 = parser.add_argument_group("exp51：Top-K 可视化")
    g_exp52 = parser.add_argument_group("exp52：瀑布图")
    g_exp53 = parser.add_argument_group("exp53：概念定位（TARIS）")
    g_exp54 = parser.add_argument_group("exp54：特征干预")
    g_exp54_tw = parser.add_argument_group("exp54：时间窗（early/late）")
    g_clip = parser.add_argument_group("exp07：CLIP 评估")

    g_main.add_argument(
        "--experiment",
        type=str,
        default="exp54",
        choices=list(SUPPORTED_EXPERIMENTS.keys()),
        help="实验编号",
    )

    # g_model.add_argument("--model_id", type=str, default="~/datasets/sd-xl/sdxl_diffusers_fp16", help="diffusers 模型目录")
    g_model.add_argument("--model_id", type=str, default="~/autodl-tmp/models/sd-xl-base-1.0-fp16-only", help="diffusers 模型目录")
    g_model.add_argument("--device", type=str, default="cuda", help="cpu 或 cuda")
    g_model.add_argument("--dtype", type=str, default="fp16", help="fp16/bf16/fp32（cpu+fp16 会自动回退）")

    # g_main.add_argument("--prompt", type=str, default="a child hold kitchen knife on the table, scary lighting.")
    # g_main.add_argument("--prompt", type=str, default="a empty street, sunny day.")
    g_main.add_argument("--prompt", type=str, default="a car in the street, sunny day.")
    g_main.add_argument("--steps", type=int, default=30)
    g_main.add_argument("--guidance_scale", type=float, default=8.0)
    g_main.add_argument("--seed", type=int, default=42)
    g_main.add_argument("--output_dir", type=str, default="./image_output")

    # g_sae.add_argument("--sae_root", type=str, default="~/sdxl-saes", help="SAE 检查点根目录")
    g_sae.add_argument("--sae_root", type=str, default="~/autodl-tmp/sdxl-saes", help="SAE 检查点根目录")
    g_sae.add_argument("--prefer_k", type=int, default=10, help="检查点选择优先 k")
    g_sae.add_argument("--prefer_hidden", type=int, default=5120, help="检查点选择优先 hidden")

    g_exp51.add_argument("--blocks",nargs="+",default=list(DEFAULT_BLOCKS),help="需要 hook 的 block 列表",)
    g_exp51.add_argument("--sae_top_k", type=int, default=10, help="top-k")
    g_exp51.add_argument("--delta_stride", type=int, default=1, help="每隔多少 step 保存一张叠加图")
    g_exp51.add_argument("--overlay_alpha", type=float, default=0.75, help="叠加透明度")
    g_exp51.add_argument(
        "--exp51_feature_csv",
        type=str,
        default="",
        help="指定特征集合的 csv 做可视化（例如 out_concept_dict/<concept>/top_positive_features.csv；留空则每步动态 top-k）",
    )
    g_exp51.add_argument(
        "--exp51_feature_k",
        type=int,
        default=10,
        help="从 csv 里取前 K 个 feature（<=0 表示全取）",
    )
    g_exp51.add_argument(
        "--exp51_feature_coeff_scale",
        type=float,
        default=1.0,
        help="固定特征集合模式下，对这些特征的系数 c_i 统一乘一个缩放（默认 1.0）",
    )
    g_exp52.add_argument("--waterfall_max_features", type=int, default=1024, help="最多画多少特征")
    g_exp52.add_argument("--waterfall_norm", type=str, default="none", choices=["row", "global", "none"], help="归一化方式")
    g_exp52.add_argument("--waterfall_cmap", type=str, default="magma", help="colormap")

    g_exp53.add_argument("--loc_block",type=str,default="unet.mid_block.attentions.0",help="用于定位概念的 SAE block",)
    g_exp53.add_argument("--concept_name",type=str,default="",help="概念名（用于组织输出目录，例如 red；留空则不加这一层）",)
    # exp53 改为从 `target_concept_dict/{concept_name}.json` 读取 prompts，
    g_exp53.add_argument("--taris_t_start", type=int, default=1000, help="时间窗口上界（高噪侧）")
    g_exp53.add_argument("--taris_t_end", type=int, default=0, help="时间窗口下界（低噪侧）")
    g_exp53.add_argument("--taris_num_steps", type=int, default=20, help="窗口内采样步数（<=0 表示全用）")
    g_exp53.add_argument("--taris_delta", type=float, default=1e-6, help="分母稳定项")
    g_exp53.add_argument("--taris_top_k", type=int, default=50, help="输出 top-k 特征数（只输出正向端）")


    g_exp54.add_argument("--int_block", type=str, 
                        # default="unet.mid_block.attentions.0",
                        default="unet.up_blocks.0.attentions.0",
                        help="要干预的 block")
    g_exp54.add_argument(
        "--targetconcept",
        type=str,
        default="car",
        help="概念名：将自动从 out_concept_dict/<targetconcept>/ 读取 csv",
    )
    g_exp54.add_argument(
        "--int_feature_top_k",
        type=int,
        default=1,
        help="从 rank_csv 取前 K 个特征",
    )

    g_exp54.add_argument("--int_mode", type=str, default="ablation", choices=["injection", "ablation"], help="injection 或 ablation")
    g_exp54.add_argument("--int_scale", type=float, default=10, help="全局强度系数（公用 scale）")
    g_exp54.add_argument(
        "--int_spatial_mask",
        type=str,
        default="none",
        choices=["none", "gaussian_center"],
        help="空间 mask（打破全图对称性）：none 或 gaussian_center",
    )
    g_exp54.add_argument(
        "--int_mask_sigma",
        type=float,
        default=0.25,
        help="gaussian_center 的 sigma 相对尺度（sigma_px = sigma * min(H,W)）",
    )
    g_exp54.add_argument("--int_t_start", type=int, default=800, help="main 窗口：t_start")
    g_exp54.add_argument("--int_t_end", type=int, default=400, help="main 窗口：t_end")
    g_exp54.add_argument("--int_step_start", type=int, default=-1, help=">=0 时启用 step 下界（优先生效）")
    g_exp54.add_argument("--int_step_end", type=int, default=-1, help=">=0 时启用 step 上界（优先生效）")
    g_exp54.add_argument("--no_baseline", action="store_true", help="不跑 baseline（节省一半计算）")

    g_exp54_tw.add_argument("--early_start", type=int, default=1000, help="early 窗口 t_start")
    g_exp54_tw.add_argument("--early_end", type=int, default=800, help="early 窗口 t_end")
    g_exp54_tw.add_argument("--late_start", type=int, default=200, help="late 窗口 t_start")
    g_exp54_tw.add_argument("--late_end", type=int, default=0, help="late 窗口 t_end")



    g_clip.add_argument("--clip_target_text", type=str, default="red")
    g_clip.add_argument("--clip_preserve_text", type=str, default="car")
    g_clip.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")

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
        exp51_feature_csv=str(args.exp51_feature_csv or ""),
        exp51_feature_k=int(args.exp51_feature_k),
        exp51_feature_coeff_scale=float(args.exp51_feature_coeff_scale),
        waterfall_max_features=args.waterfall_max_features,
        waterfall_norm=args.waterfall_norm,
        waterfall_cmap=args.waterfall_cmap,
    )

    # 干预特征参数统一口径：
    # - 由 exp54 内部从 rank_csv 取 top-k

    # 约定：<0 表示“禁用该 step 边界”
    step_start = None if int(args.int_step_start) < 0 else int(args.int_step_start)
    step_end = None if int(args.int_step_end) < 0 else int(args.int_step_end)
    # 从 out_concept_dict/<concept>/ 取 csv
    targetconcept = str(getattr(args, "targetconcept", "") or "").strip()
    if not targetconcept:
        raise ValueError("--targetconcept 不能为空。")
    base_dir = os.path.join("out_concept_dict", targetconcept)
    coeff_csv = os.path.join(base_dir, "feature_time_scores.csv")

    int_cfg = CausalInterventionConfig(
        block=args.int_block,
        feature_top_k=int(args.int_feature_top_k),
        mode=args.int_mode,
        scale=float(args.int_scale),
        spatial_mask=str(args.int_spatial_mask),
        mask_sigma=float(args.int_mask_sigma),
        coeff_csv=str(coeff_csv),
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
