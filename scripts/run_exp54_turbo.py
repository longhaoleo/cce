#!/usr/bin/env python3
"""
SDXL-Turbo 下的 exp54 快速验证脚本（全程干预）。

目标：
- 用 Turbo 的短步数快速验证“干预是否有效”；
- 概念特征直接复用 exp53 输出（out_concept_dict_<block_short>/<concept>/...）；
- 默认全程干预（覆盖全部推理 step）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """补齐导入路径，确保从任意工作目录可运行。"""
    repo_root = Path(__file__).resolve().parents[1]
    scripts_root = repo_root / "scripts"
    for p in (str(repo_root), str(scripts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_path()

from sdxl_wsae.configs import (  # noqa: E402
    CausalInterventionConfig,
    ModelConfig,
    RunConfig,
    SAEConfig,
)
from sdxl_wsae.experiments.exp54_intervention_suite import run_exp54_causal_intervention  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run exp54 on SDXL-Turbo with full-trajectory intervention")
    g_main = ap.add_argument_group("主参数")
    g_model = ap.add_argument_group("模型/SAE")
    g_run = ap.add_argument_group("采样")
    g_int = ap.add_argument_group("干预")

    g_main.add_argument("--output_dir", type=str, default="./wsae_res_sdxl_turbo_output")
    g_main.add_argument("--targetconcept", type=str, default="cyberpunk", help="exp53 的概念名（对应概念目录名）")

    g_model.add_argument("--model_id", type=str, default="~/datasets/sd-xl-turbo")
    g_model.add_argument("--device", type=str, default="cuda")
    g_model.add_argument("--dtype", type=str, default="fp16")
    g_model.add_argument("--sae_root", type=str, default="~/sdxl-saes")
    g_model.add_argument("--prefer_k", type=int, default=10)
    g_model.add_argument("--prefer_hidden", type=int, default=5120)

    g_run.add_argument("--prompt", type=str, default="a car in the street, sunny day.")

    g_run.add_argument("--steps", type=int, default=4, help="Turbo 建议短步数（默认 4）")
    g_run.add_argument("--guidance_scale", type=float, default=0.0, help="Turbo 常用 0.0，可按需调整")
    g_run.add_argument("--seed", type=int, default=42)

    g_int.add_argument(
        "--blocks",
        nargs="+",
        type=str,
        default=[
            "unet.down_blocks.2.attentions.1",
            "unet.mid_block.attentions.0",
            "unet.up_blocks.0.attentions.0",
            "unet.up_blocks.0.attentions.1",
            ],
        help="可传多个 block，同时 hook 干预",
    )
    g_int.add_argument("--int_mode", type=str, default="injection", choices=["injection", "ablation"])
    g_int.add_argument("--int_scale", type=float, default=10.0, help="全局干预强度")
    g_int.add_argument("--int_feature_top_k", type=int, default=5, help="从 exp53 的 top_positive_features.csv 取前 K")
    g_int.add_argument(
        "--int_use_time_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用 exp53 的时间权重",
    )
    g_int.add_argument(
        "--int_use_spatial_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用空间归一化权重",
    )
    g_int.add_argument(
        "--int_spatial_mask",
        type=str,
        default="none",
        choices=["none", "gaussian_center"],
    )
    g_int.add_argument("--int_mask_sigma", type=float, default=0.25)
    g_int.add_argument("--no_baseline", action="store_true", help="不生成 baseline（更快）")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = ModelConfig(
        model_id=str(args.model_id),
        device=str(args.device),
        dtype_name=str(args.dtype),
    )
    sae_cfg = SAEConfig(
        sae_root=str(args.sae_root),
        blocks=tuple(str(b) for b in args.blocks),
        prefer_k=int(args.prefer_k),
        prefer_hidden=int(args.prefer_hidden),
    )
    run_cfg = RunConfig(
        prompt=str(args.prompt),
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
    )

    # 全程干预：时间窗覆盖 [0, 1000]，step 窗口覆盖所有 step。
    int_cfg = CausalInterventionConfig(
        blocks=tuple(str(b) for b in args.blocks),
        targetconcept=str(args.targetconcept),
        feature_top_k=int(args.int_feature_top_k),
        mode=str(args.int_mode),
        scale=float(args.int_scale),
        use_time_weight=bool(args.int_use_time_weight),
        use_spatial_norm_weight=bool(args.int_use_spatial_weight),
        spatial_mask=str(args.int_spatial_mask),
        mask_sigma=float(args.int_mask_sigma),
        t_start=1000,
        t_end=0,
        step_start=0,
        step_end=max(0, int(args.steps) - 1),
        compare_baseline=(not bool(args.no_baseline)),
    )

    out_dir = os.path.expanduser(str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)
    run_exp54_causal_intervention(
        model_cfg=model_cfg,
        sae_cfg=sae_cfg,
        run_cfg=run_cfg,
        int_cfg=int_cfg,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()

