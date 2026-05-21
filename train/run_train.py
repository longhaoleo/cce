#!/usr/bin/env python3
"""
Shared SAE 训练入口脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def _bootstrap_path() -> None:
    """补齐导入路径，允许 `python train/run_train.py` 直接运行。

    输入：
    - 无

    输出：
    - 无；原地修改 `sys.path`。
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()


def _sanitize_omp_num_threads() -> None:
    """清理非法的 OMP_NUM_THREADS，避免 torch/libgomp 启动时报错。"""
    raw = os.environ.get("OMP_NUM_THREADS", "")
    if not raw:
        return
    try:
        if int(raw) <= 0:
            raise ValueError
    except Exception:
        print(f"[train] 检测到非法 OMP_NUM_THREADS={raw!r}，已自动忽略。")
        os.environ.pop("OMP_NUM_THREADS", None)


_sanitize_omp_num_threads()

from SAE import SharedSAE, TrainConfig, estimate_block_scales  # noqa: E402
from train.metrics import plot_loss_curves  # noqa: E402
from train.prompt_data import (  # noqa: E402
    load_prompts_from_csv,
    maybe_truncate,
    split_prompt_records,
    summarize_split,
)
from train.sampler import SDXLGroupSampler  # noqa: E402
from train.trainer import SharedSAETrainer  # noqa: E402


EXPERIMENT_PRESETS = (
    "custom",
    "exp_a_shared_recon",
    "exp_b_shared_align",
    "exp_c_adapter_align",
    "exp_d_full",
)


def parse_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser(
        description="Shared SAE 主训练入口（stage2 -> stage3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_io = ap.add_argument_group("输入输出")
    g_io.add_argument(
        "--prompts_csv",
        type=str,
        default="data/coco_30k.csv",
        help="Prompt 数据 CSV 路径；优先读取 `prompt` 列，若缺失则读取首列。",
    )
    g_io.add_argument(
        "--output_root",
        type=str,
        default="output_shared_sae",
        help="训练输出根目录（checkpoint、run_manifest、阶段日志都会写到这里）。",
    )
    g_io.add_argument(
        "--experiment_preset",
        type=str,
        default="custom",
        choices=list(EXPERIMENT_PRESETS),
        help="渐进实验预设；用一个名字替代一串手动开关。",
    )

    g_runtime = ap.add_argument_group("运行与采样")
    g_runtime.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Diffusers 模型标识；默认使用 SDXL Base。",
    )
    g_runtime.add_argument(
        "--model_local_dir",
        type=str,
        default="/root/autodl-tmp/models/sd-xl-base-1.0-fp16-only",
        help="本地模型目录（优先级最高）。若填写则强制从本地目录离线加载。",
    )
    g_runtime.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否仅使用本地缓存，不访问网络。",
    )
    g_runtime.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备，常用 `cuda` 或 `cpu`。",
    )
    g_runtime.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="模型推理精度：`fp16/bf16/fp32`。会影响显存和速度。",
    )
    g_runtime.add_argument(
        "--steps",
        type=int,
        default=30,
        help="每条 prompt 的扩散步数。步数越大，采样越慢。",
    )
    g_runtime.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="CFG guidance 强度。",
    )
    g_runtime.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="目标图像分辨率；默认使用当前 Shared 主线的 512-space 基线。",
    )
    g_runtime.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="基础随机种子；若 CSV 行内没有 seed，则使用 `base_seed + row_idx`。",
    )
    g_runtime.add_argument(
        "--split_seed",
        type=int,
        default=2026,
        help="数据集切分随机种子（控制 train/val/calibration 划分可复现）。",
    )
    g_runtime.add_argument(
        "--max_prompts_debug",
        type=int,
        default=0,
        help="调试截断数量；>0 时只使用前 N 条 prompt。",
    )

    g_data = ap.add_argument_group("数据划分与采样粒度")
    g_data.add_argument(
        "--validation_prompts",
        type=int,
        default=1000,
        help="验证集 prompt 数量（与训练集严格不重叠）。",
    )
    g_data.add_argument(
        "--stage2_train_prompts",
        type=int,
        default=20000,
        help="Stage2/Stage3 训练集大小。",
    )
    g_data.add_argument(
        "--calibration_prompts",
        type=int,
        default=1000,
        help="用于估计 block 归一化系数 s_b 的校准集大小。",
    )
    g_data.add_argument(
        "--num_step_buckets",
        type=int,
        default=5,
        help="时间桶数量；每条 prompt 每桶随机取 1 个 step。",
    )
    g_data.add_argument(
        "--shard_prompts",
        type=int,
        default=250,
        help="每个 prompt 分片的大小；影响 CPU 内存占用和采样批次时长。",
    )

    g_model = ap.add_argument_group("SAE 主结构")
    g_model.add_argument(
        "--expansion_factor",
        type=int,
        default=4,
        help="字典扩展倍数；`n_dirs = expansion_factor * d_model`。",
    )
    g_model.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K 稀疏激活数（每个 token 保留的激活特征个数）。",
    )
    g_model.add_argument(
        "--auxk",
        type=int,
        default=256,
        help="AuxK 辅助分支的 top-k 数量。",
    )
    g_model.add_argument(
        "--dead_tokens_threshold",
        type=int,
        default=10_000_000,
        help="判定 dead feature 的累计未激活 token 阈值。",
    )

    g_cond = ap.add_argument_group("时间/空间条件分支")
    g_cond.add_argument(
        "--use_time_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用时间分支；做简化 baseline 时可关闭。",
    )
    g_cond.add_argument(
        "--time_branch_mode",
        type=str,
        default="sincos_linear",
        choices=["sincos_linear", "sincos_mlp", "sincos_film"],
        help="时间分支模式：linear/mlp/film。",
    )
    g_cond.add_argument(
        "--use_spatial_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用空间分支；做简化 baseline 时可关闭。",
    )
    g_cond.add_argument(
        "--spatial_branch_mode",
        type=str,
        default="sincos_linear",
        choices=["sincos_linear", "sincos_mlp", "sincos_film"],
        help="空间分支模式：linear/mlp/film。",
    )
    g_cond.add_argument(
        "--time_embed_dim",
        type=int,
        default=32,
        help="时间 1D 正余弦编码维度。",
    )
    g_cond.add_argument(
        "--time_hidden_dim",
        type=int,
        default=128,
        help="时间分支 MLP 隐层维度（linear 模式下可忽略）。",
    )
    g_cond.add_argument(
        "--time_branch_warmup_start_ratio",
        type=float,
        default=0.0,
        help="time branch 延迟开启比例；0 表示从阶段开始就可用。",
    )
    g_cond.add_argument(
        "--time_branch_warmup_ratio",
        type=float,
        default=0.0,
        help="time branch 从 0 线性升到 1 的阶段比例；0 表示不做调度。",
    )
    g_cond.add_argument(
        "--spatial_embed_dim",
        type=int,
        default=64,
        help="空间 2D 正余弦编码维度（需能被 4 整除）。",
    )
    g_cond.add_argument(
        "--spatial_hidden_dim",
        type=int,
        default=128,
        help="空间分支 MLP 隐层维度（linear 模式下可忽略）。",
    )

    g_adapter = ap.add_argument_group("LoRA 适配器")
    g_adapter.add_argument(
        "--block_in_rank",
        type=int,
        default=16,
        help="输入适配器 LoRA rank。",
    )
    g_adapter.add_argument(
        "--block_in_alpha",
        type=int,
        default=16,
        help="输入适配器 LoRA alpha（缩放系数分子）。",
    )
    g_adapter.add_argument(
        "--use_block_in_adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 block_in_adapter。",
    )
    g_adapter.add_argument(
        "--run_stage3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否执行 stage3 微调；做快速对比实验时可关闭。",
    )

    g_log = ap.add_argument_group("日志与保存")
    g_log.add_argument(
        "--log_every_steps",
        type=int,
        default=20,
        help="每多少个优化 step 打印一次训练日志。",
    )
    g_log.add_argument(
        "--save_every_steps",
        type=int,
        default=200,
        help="每多少个优化 step 保存一次 checkpoint；传 0 表示关闭中间 checkpoint，仅保留阶段末尾保存。",
    )

    g_opt = ap.add_argument_group("损失与 batch 组织")
    g_opt.add_argument(
        "--lr_time",
        type=float,
        default=1e-4,
        help="stage2 中 time branch 学习率。",
    )
    g_opt.add_argument(
        "--lr_time_stage3",
        type=float,
        default=2e-5,
        help="stage3 中 time branch 学习率。",
    )
    g_opt.add_argument(
        "--align_weight_target",
        type=float,
        default=5e-2,
        help="Stage2/3 对齐损失目标权重。",
    )
    g_opt.add_argument(
        "--align_warmup_ratio",
        type=float,
        default=0.1,
        help="Stage2 中 align 权重 warmup 占比。",
    )
    g_opt.add_argument(
        "--latent_decorr_weight",
        type=float,
        default=0.0,
        help="latent 协方差去相关正则权重；直接惩罚 batch 内 feature 共激活。",
    )
    g_opt.add_argument(
        "--latent_decorr_top_k",
        type=int,
        default=256,
        help="latent 去相关只作用于 batch 内 top-active feature；0 表示关闭。",
    )
    g_opt.add_argument(
        "--latent_decorr_mode",
        type=str,
        default="token",
        choices=["token", "block_pooled"],
        help="latent 去相关的样本组织方式。",
    )
    g_opt.add_argument(
        "--latent_decorr_pool",
        type=str,
        default="mean",
        choices=["mean", "topq", "hybrid"],
        help="block_pooled 模式下的 token pooling 方式。",
    )
    g_opt.add_argument(
        "--latent_decorr_pool_topq",
        type=float,
        default=0.1,
        help="topq/hybrid pooling 使用的 token top fraction。",
    )
    g_opt.add_argument(
        "--latent_decorr_eps",
        type=float,
        default=1e-4,
        help="latent 去相关标准化的数值稳定项。",
    )
    g_opt.add_argument(
        "--tokens_per_step_target",
        type=int,
        default=4096,
        help="用于自动推导 group batch size 的目标 token 数。",
    )
    g_opt.add_argument(
        "--group_bs",
        type=int,
        default=0,
        help="Stage2/3 的 group batch size；传 0 表示按真实 hw 自动推导。",
    )
    return ap.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """由 CLI 参数构建训练配置对象。

    输入：
    - args: 命令行解析结果。

    输出：
    - TrainConfig：完整训练配置。
    """
    cfg = TrainConfig(
        prompts_csv=str(args.prompts_csv),
        output_root=str(args.output_root),
        experiment_preset=str(args.experiment_preset),
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=str(args.device),
        dtype=str(args.dtype),
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        resolution=int(args.resolution),
        base_seed=int(args.base_seed),
        split_seed=int(args.split_seed),
        max_prompts_debug=int(args.max_prompts_debug),
        validation_prompts=int(args.validation_prompts),
        stage2_train_prompts=int(args.stage2_train_prompts),
        calibration_prompts=int(args.calibration_prompts),
        num_step_buckets=int(args.num_step_buckets),
        shard_prompts=int(args.shard_prompts),
        expansion_factor=int(args.expansion_factor),
        top_k=int(args.top_k),
        auxk=int(args.auxk),
        dead_tokens_threshold=int(args.dead_tokens_threshold),
        use_time_branch=bool(args.use_time_branch),
        time_branch_mode=str(args.time_branch_mode),
        use_spatial_branch=bool(args.use_spatial_branch),
        spatial_branch_mode=str(args.spatial_branch_mode),
        time_embed_dim=int(args.time_embed_dim),
        time_hidden_dim=int(args.time_hidden_dim),
        time_branch_warmup_start_ratio=float(args.time_branch_warmup_start_ratio),
        time_branch_warmup_ratio=float(args.time_branch_warmup_ratio),
        spatial_embed_dim=int(args.spatial_embed_dim),
        spatial_hidden_dim=int(args.spatial_hidden_dim),
        lr_time=float(args.lr_time),
        lr_time_stage3=float(args.lr_time_stage3),
        align_weight_target=float(args.align_weight_target),
        align_warmup_ratio=float(args.align_warmup_ratio),
        latent_decorr_weight=float(args.latent_decorr_weight),
        latent_decorr_top_k=int(args.latent_decorr_top_k),
        latent_decorr_mode=str(args.latent_decorr_mode),
        latent_decorr_pool=str(args.latent_decorr_pool),
        latent_decorr_pool_topq=float(args.latent_decorr_pool_topq),
        latent_decorr_eps=float(args.latent_decorr_eps),
        tokens_per_step_target=int(args.tokens_per_step_target),
        group_bs=int(args.group_bs),
        block_in_rank=int(args.block_in_rank),
        block_in_alpha=int(args.block_in_alpha),
        use_block_in_adapter=bool(args.use_block_in_adapter),
        run_stage3=bool(args.run_stage3),
        log_every_steps=int(args.log_every_steps),
        save_every_steps=int(args.save_every_steps),
    )
    apply_experiment_preset(cfg, cfg.experiment_preset)
    cfg.validate()
    cfg.ensure_paths()
    return cfg


def apply_experiment_preset(cfg: TrainConfig, preset: str) -> None:
    """按实验预设覆盖一组渐进实验配置。

    输入：
    - cfg: 训练配置对象。
    - preset: 预设名称。

    输出：
    - 无；原地修改 cfg。
    """
    name = str(preset).strip()
    cfg.experiment_preset = name
    if name == "custom":
        return
    if name == "exp_a_shared_recon":
        cfg.run_stage3 = False
        cfg.align_weight_target = 0.0
        cfg.use_block_in_adapter = False
        cfg.use_block_out_adapter = False
        cfg.use_time_branch = False
        cfg.use_spatial_branch = False
        return
    if name == "exp_b_shared_align":
        cfg.run_stage3 = False
        cfg.align_weight_target = 5e-2
        cfg.use_block_in_adapter = False
        cfg.use_block_out_adapter = False
        cfg.use_time_branch = False
        cfg.use_spatial_branch = False
        return
    if name == "exp_c_adapter_align":
        cfg.run_stage3 = True
        cfg.align_weight_target = 5e-2
        cfg.use_block_in_adapter = True
        cfg.use_block_out_adapter = False
        cfg.use_time_branch = False
        cfg.use_spatial_branch = False
        return
    if name == "exp_d_full":
        cfg.run_stage3 = True
        cfg.align_weight_target = 5e-2
        cfg.use_block_in_adapter = True
        cfg.use_block_out_adapter = False
        cfg.use_time_branch = True
        cfg.use_spatial_branch = True
        return
    raise ValueError(f"未知 experiment_preset: {name}")


def save_run_manifest(output_root: str, payload: Dict) -> None:
    """保存运行元信息文件。

    输入：
    - output_root: 输出目录。
    - payload: 要写入 JSON 的字典。

    输出：
    - 无；写入 `run_manifest.json`。
    """
    out = Path(output_root).expanduser().resolve() / "run_manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """主入口：执行完整四阶段训练流程。

    输入：
    - 无（从 CLI 读取配置）。

    输出：
    - 无；在 `output_root` 生成 checkpoint、配置和日志文件。
    """
    args = parse_args()
    cfg = build_config(args)
    if cfg.experiment_preset != "custom":
        print(f"[train] 应用实验预设: {cfg.experiment_preset}")

    print("[train] 读取 prompts ...")
    all_records = load_prompts_from_csv(cfg.prompts_csv, base_seed=cfg.base_seed)
    all_records = maybe_truncate(all_records, cfg.max_prompts_debug)
    split = split_prompt_records(
        all_records,
        split_seed=cfg.split_seed,
        validation_prompts=cfg.validation_prompts,
        stage2_train_prompts=cfg.stage2_train_prompts,
        calibration_prompts=cfg.calibration_prompts,
    )
    print(f"[train] split 统计: {summarize_split(split)}")

    print("[train] 初始化采样器 ...")
    sampler = SDXLGroupSampler(
        model_id=cfg.model_id,
        model_local_dir=cfg.model_local_dir,
        local_files_only=cfg.local_files_only,
        device=cfg.device,
        dtype_name=cfg.dtype,
        steps=cfg.steps,
        guidance_scale=cfg.guidance_scale,
        num_step_buckets=cfg.num_step_buckets,
        resolution=cfg.resolution,
        expected_h=cfg.expected_h,
        expected_w=cfg.expected_w,
    )

    print("[train] 估计 block 归一化系数 s_b ...")
    norm_scale_by_block = estimate_block_scales(
        sampler=sampler,
        calibration_records=split["calibration"],
        blocks=cfg.blocks,
        d_model=cfg.d_model,
    )
    print(f"[train] norm_scale_by_block={norm_scale_by_block}")

    print("[train] 构建 Shared SAE 模型 ...")
    model = SharedSAE(
        blocks=cfg.blocks,
        d_model=cfg.d_model,
        n_dirs=cfg.n_dirs,
        top_k=cfg.top_k,
        auxk=cfg.auxk,
        dead_tokens_threshold=cfg.dead_tokens_threshold,
        use_block_in_adapter=cfg.use_block_in_adapter,
        use_block_out_adapter=cfg.use_block_out_adapter,
        block_in_rank=cfg.block_in_rank,
        block_in_alpha=cfg.block_in_alpha,
        block_out_rank=cfg.block_out_rank,
        block_out_alpha=cfg.block_out_alpha,
        use_time_branch=cfg.use_time_branch,
        time_branch_mode=cfg.time_branch_mode,
        time_embed_dim=cfg.time_embed_dim,
        time_hidden_dim=cfg.time_hidden_dim,
        use_spatial_branch=cfg.use_spatial_branch,
        spatial_branch_mode=cfg.spatial_branch_mode,
        spatial_embed_dim=cfg.spatial_embed_dim,
        spatial_hidden_dim=cfg.spatial_hidden_dim,
    )

    trainer = SharedSAETrainer(
        cfg=cfg,
        model=model,
        sampler=sampler,
        norm_scale_by_block=norm_scale_by_block,
    )

    stages = [("stage2", split["stage2"])]
    if bool(cfg.run_stage3):
        stages.append(("stage3", split["stage2"]))

    stage_results = []
    for stage, records in stages:
        print(f"[train] 开始 {stage} ...")
        res = trainer.run_stage(stage, records)
        print(
            f"[train] {stage} 完成: steps={res.steps} "
            f"loss={res.mean_total:.6f} recon={res.mean_recon:.6f} "
            f"auxk={res.mean_auxk:.6f} align={res.mean_align:.6f} "
            f"time={res.elapsed_sec:.1f}s"
        )
        val_metrics = trainer.evaluate_stage_metrics(split["validation"], stage=stage, max_groups=120)
        print(
            f"[train] {stage} 验证 recon={val_metrics.recon:.6f} "
            f"align={val_metrics.align:.6f} groups={val_metrics.groups}"
        )
        stage_payload = dict(res.__dict__)
        stage_payload["val_recon"] = float(val_metrics.recon)
        stage_payload["val_align"] = float(val_metrics.align)
        stage_payload["val_groups"] = int(val_metrics.groups)
        stage_results.append(stage_payload)

    curve_paths = plot_loss_curves(cfg.output_root)
    if curve_paths:
        print(f"[train] loss 曲线: {curve_paths.get('loss_curves', '')}")
        print(f"[train] align 曲线: {curve_paths.get('align_weight_curve', '')}")
    else:
        print("[train] 跳过曲线绘图（可能缺少 matplotlib 或无 step 指标）。")

    save_run_manifest(
        cfg.output_root,
        payload={
            "config": cfg.to_dict(),
            "norm_scale_by_block": norm_scale_by_block,
            "split_sizes": summarize_split(split),
            "stage_results": stage_results,
            "curve_paths": curve_paths,
        },
    )
    print(f"[train] 全部完成，输出目录: {Path(cfg.output_root).resolve()}")


if __name__ == "__main__":
    main()
