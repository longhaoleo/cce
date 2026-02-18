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
from .configs import TemporalWindowConfig


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数分层：
    - 通用层：模型路径、采样参数、输出路径；
    - 可视化层：exp51 / exp52 参数；
    - 干预层：exp54/05/06/07/21 的 feature intervention 参数；
    - 评估层：exp07 的 CLIP 文本参数；
    - 时间层：exp54 的早晚注入窗口。
    """
    parser = argparse.ArgumentParser(description="SDXL SAE 实验平台（统一入口）")

    g_main = parser.add_argument_group("主参数（通用）")
    g_model = parser.add_argument_group("模型参数（SDXL）")
    g_sae = parser.add_argument_group("SAE 参数（通用）")
    g_viz = parser.add_argument_group("可视化参数（exp51/exp52）")
    g_int = parser.add_argument_group("干预参数（exp54/exp05/exp06/exp07）")
    g_tw = parser.add_argument_group("时间窗参数（exp54 early/late）")
    g_clip = parser.add_argument_group("CLIP 参数（exp07）")
    g_loc = parser.add_argument_group("概念定位（exp53 TARIS）")

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
    g_main.add_argument("--output_dir", type=str, default="./wsae_res_sdxl_output")

    # g_sae.add_argument("--sae_root", type=str, default="~/sdxl-saes", help="SAE 检查点根目录")
    g_sae.add_argument("--sae_root", type=str, default="~/autodl-tmp/sdxl-saes", help="SAE 检查点根目录")
    g_sae.add_argument("--prefer_k", type=int, default=10, help="检查点选择优先 k")
    g_sae.add_argument("--prefer_hidden", type=int, default=5120, help="检查点选择优先 hidden")
    g_viz.add_argument(
        "--blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        help="exp51/52 用：需要 hook 的 block 列表",
    )

    g_viz.add_argument("--sae_top_k", type=int, default=10, help="exp51 用：top-k")
    g_viz.add_argument("--delta_stride", type=int, default=1, help="exp51 用：每隔多少 step 保存一张叠加图")
    g_viz.add_argument("--overlay_alpha", type=float, default=0.75, help="exp51 用：叠加透明度")
    g_viz.add_argument(
        "--exp51_feature_csv",
        type=str,
        default="",
        help="exp51 用：指定特征集合的 csv（例如 out_concept_dict/<concept>/top_positive_features.csv；留空则每步动态 top-k）",
    )
    g_viz.add_argument(
        "--exp51_feature_k",
        type=int,
        default=0,
        help="exp51 用：从 csv 里取前 K 个 feature（<=0 表示全取）",
    )
    g_viz.add_argument(
        "--exp51_feature_coeff_scale",
        type=float,
        default=1.0,
        help="exp51 用：固定特征集合模式下，对这些特征的系数 c_i 统一乘一个缩放（默认 1.0）",
    )
    g_viz.add_argument("--waterfall_max_features", type=int, default=1024, help="exp52 用：最多画多少特征")
    g_viz.add_argument("--waterfall_norm", type=str, default="row", choices=["row", "global", "none"], help="exp52 用：归一化方式")
    g_viz.add_argument("--waterfall_cmap", type=str, default="magma", help="exp52 用：colormap")

    g_int.add_argument("--int_block", type=str, default="unet.mid_block.attentions.0", help="要干预的 block")
    g_int.add_argument(
        "--int_feature_ids",
        nargs="+",
        type=int,
        default=[2758, 4052, 919, 473, 366, 878, 2229, 2215, 2932, 2091],
        help="干预特征 id 列表（单特征就传 1 个）",
    )
    g_int.add_argument(
        "--int_feature_scales",
        nargs="+",
        type=float,
        default=[0.006224330514669418, 0.005031862761825323, 0.00404663709923625, 0.00349993584677577, 
                 0.002999882912263274, 0.002881488762795925, 0.002654273994266987, 0.0024882021825760603, 
                 0.0024667761754244566, 0.0023173089139163494],
        help="每个特征的相对系数（可选；可只给 1 个值用于广播；最终强度=int_scale*feature_scale）",
    )

    g_int.add_argument("--int_mode", type=str, default="ablation", choices=["injection", "ablation"], help="injection 或 ablation")
    g_int.add_argument("--int_scale", type=float, default=10000, help="全局强度系数（公用 scale）")
    g_int.add_argument(
        "--int_spatial_mask",
        type=str,
        default="gaussian_center",
        choices=["none", "gaussian_center"],
        help="空间 mask（打破全图对称性）：none 或 gaussian_center",
    )
    g_int.add_argument(
        "--int_mask_sigma",
        type=float,
        default=0.25,
        help="gaussian_center 的 sigma 相对尺度（sigma_px = sigma * min(H,W)）",
    )
    g_int.add_argument("--int_t_start", type=int, default=900, help="main 窗口：t_start")
    g_int.add_argument("--int_t_end", type=int, default=600, help="main 窗口：t_end")
    g_int.add_argument("--int_step_start", type=int, default=-1, help=">=0 时启用 step 下界（优先生效）")
    g_int.add_argument("--int_step_end", type=int, default=-1, help=">=0 时启用 step 上界（优先生效）")
    g_int.add_argument("--no_baseline", action="store_true", help="不跑 baseline（节省一半计算）")

    g_clip.add_argument("--clip_target_text", type=str, default="red")
    g_clip.add_argument("--clip_preserve_text", type=str, default="car")
    g_clip.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")

    g_tw.add_argument("--early_start", type=int, default=1000, help="early 窗口 t_start")
    g_tw.add_argument("--early_end", type=int, default=800, help="early 窗口 t_end")
    g_tw.add_argument("--late_start", type=int, default=200, help="late 窗口 t_start")
    g_tw.add_argument("--late_end", type=int, default=0, help="late 窗口 t_end")

    g_loc.add_argument("--loc_block",type=str,default="unet.mid_block.attentions.0",help="exp53 用：用于定位概念的 SAE block",)
    g_loc.add_argument("--concept_name",type=str,default="",help="exp53 用：概念名（用于组织输出目录，例如 red；留空则不加这一层）",)
    # exp53 改为从 `target_concept_dict/{concept_name}.json` 读取 prompts，
    # 所以不再从 CLI 接收超长的 pos/neg 列表参数。
    g_loc.add_argument("--taris_t_start", type=int, default=800, help="时间窗口上界（高噪侧）")
    g_loc.add_argument("--taris_t_end", type=int, default=200, help="时间窗口下界（低噪侧）")
    g_loc.add_argument("--taris_num_steps", type=int, default=10, help="窗口内采样步数（<=0 表示全用）")
    g_loc.add_argument("--taris_delta", type=float, default=1e-6, help="分母稳定项")
    g_loc.add_argument("--taris_top_k", type=int, default=20, help="输出 top-k 特征数（只输出正向端）")

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
    # - 只保留 --int_feature_ids（传 1 个就是单特征）
    # - feature_scales 可不传（默认全 1）；也可只传 1 个数（自动广播）；或传与 ids 等长的列表
    feature_ids = [int(x) for x in (args.int_feature_ids or [])]
    if not feature_ids:
        raise ValueError("--int_feature_ids 不能为空。")

    if args.int_feature_scales:
        scales = [float(x) for x in args.int_feature_scales]
        if len(scales) == 1 and len(feature_ids) > 1:
            scales = [scales[0] for _ in feature_ids]
        if len(scales) != len(feature_ids):
            raise ValueError("--int_feature_scales 长度必须与 --int_feature_ids 相同（或只给 1 个用于广播）。")
        feature_scales = scales
    else:
        feature_scales = [1.0 for _ in feature_ids]

    # 约定：<0 表示“禁用该 step 边界”
    step_start = None if int(args.int_step_start) < 0 else int(args.int_step_start)
    step_end = None if int(args.int_step_end) < 0 else int(args.int_step_end)
    int_cfg = CausalInterventionConfig(
        block=args.int_block,
        feature_ids=tuple(feature_ids),
        feature_scales=tuple(feature_scales),
        mode=args.int_mode,
        scale=float(args.int_scale),
        spatial_mask=str(args.int_spatial_mask),
        mask_sigma=float(args.int_mask_sigma),
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
