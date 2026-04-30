#!/usr/bin/env python3
"""
Smoke 训练入口：用于快速验证训练链路是否通畅。

说明：
- 该脚本不会改动 `run_train.py` 的主逻辑；
- 通过传入一组小规模参数，调用正式训练入口；
- 适合先做 5~30 分钟级别的冒烟验证。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _sanitize_omp_env(env: dict[str, str]) -> dict[str, str]:
    """移除非法 OMP_NUM_THREADS，避免 smoke 子进程启动时报错。"""
    out = dict(env)
    raw = out.get("OMP_NUM_THREADS", "")
    if not raw:
        return out
    try:
        if int(raw) <= 0:
            raise ValueError
    except Exception:
        print(f"[smoke] 检测到非法 OMP_NUM_THREADS={raw!r}，子进程中将忽略该变量。")
        out.pop("OMP_NUM_THREADS", None)
    return out


def parse_args() -> argparse.Namespace:
    """解析 smoke 训练参数。

    输入：
    - 无（从命令行读取）。

    输出：
    - argparse.Namespace：smoke 参数集合。
    """
    ap = argparse.ArgumentParser(
        description="Shared SAE smoke train runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--prompts_csv", type=str, default="data/coco_30k.csv", help="prompt 数据源 CSV。")
    ap.add_argument("--output_root", type=str, default="train/output_shared_sae_smoke", help="smoke 输出目录。")
    ap.add_argument(
        "--experiment_preset",
        type=str,
        default="custom",
        choices=["custom", "exp_a_shared_recon", "exp_b_shared_align", "exp_c_adapter_align", "exp_d_full"],
        help="渐进实验预设；会透传给正式训练入口。",
    )
    ap.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="模型标识。")
    ap.add_argument("--model_local_dir", type=str, default="", help="本地模型目录（若填则离线优先）。")
    ap.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否仅使用本地缓存，不访问网络。",
    )
    ap.add_argument("--device", type=str, default="cuda", help="运行设备。")
    ap.add_argument("--dtype", type=str, default="fp16", help="推理精度。")
    ap.add_argument("--steps", type=int, default=20, help="smoke 采样步数。")
    ap.add_argument("--guidance_scale", type=float, default=8.0, help="CFG guidance。")
    ap.add_argument("--resolution", type=int, default=512, help="采样分辨率；默认使用当前 Shared 主线的 512-space 基线。")

    ap.add_argument("--validation_prompts", type=int, default=20, help="验证集大小。")
    ap.add_argument("--stage2_train_prompts", type=int, default=80, help="Stage2/3 训练集大小。")
    ap.add_argument("--stage1_train_prompts", type=int, default=20, help="Stage1 训练集大小。")
    ap.add_argument("--calibration_prompts", type=int, default=20, help="归一化校准集大小。")
    ap.add_argument("--num_step_buckets", type=int, default=2, help="每条 prompt 抽样时间桶数量。")
    ap.add_argument("--shard_prompts", type=int, default=10, help="每个 shard 的 prompt 数。")
    ap.add_argument("--max_prompts_debug", type=int, default=120, help="总样本上限（调试）。")

    ap.add_argument("--expansion_factor", type=int, default=2, help="字典扩展倍数（smoke 默认更小）。")
    ap.add_argument("--top_k", type=int, default=8, help="Top-K 激活数。")
    ap.add_argument("--auxk", type=int, default=64, help="AuxK 数量。")
    ap.add_argument("--dead_tokens_threshold", type=int, default=10_000_000, help="dead feature 阈值。")
    ap.add_argument(
        "--use_block_in_adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 block_in_adapter。",
    )
    ap.add_argument("--align_weight_target", type=float, default=5e-2, help="align 目标权重。")
    ap.add_argument("--align_warmup_ratio", type=float, default=0.1, help="align warmup 占比。")
    ap.add_argument("--tokens_per_step_target", type=int, default=4096, help="自动推导 group_bs 的目标 token 数。")
    ap.add_argument("--group_bs_stage1", type=int, default=0, help="Stage1 group batch size；0 表示自动推导。")
    ap.add_argument("--group_bs_stage2", type=int, default=0, help="Stage2/3/4 group batch size；0 表示自动推导。")
    ap.add_argument(
        "--run_stage1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否执行 stage1 预热；默认开启。",
    )
    ap.add_argument(
        "--run_stage3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否执行 stage3 微调；默认开启。",
    )
    ap.add_argument(
        "--run_stage4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在 smoke 里追加 stage4 out_adapter 探针；默认关闭。",
    )
    ap.add_argument(
        "--use_time_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用时间分支。",
    )
    ap.add_argument(
        "--time_branch_mode",
        type=str,
        default="sincos_linear",
        choices=["sincos_linear", "sincos_mlp", "sincos_film"],
        help="时间分支模式。",
    )
    ap.add_argument(
        "--use_spatial_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用空间分支。",
    )
    ap.add_argument(
        "--spatial_branch_mode",
        type=str,
        default="sincos_linear",
        choices=["sincos_linear", "sincos_mlp", "sincos_film"],
        help="空间分支模式。",
    )

    ap.add_argument("--save_every_steps", type=int, default=20, help="checkpoint 保存频率。")
    ap.add_argument("--log_every_steps", type=int, default=1, help="日志打印频率。")

    ap.add_argument(
        "--print_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅打印将执行的命令，不真正启动训练。",
    )
    ap.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="透传给 run_train.py 的额外参数。用法：`-- --arg1 xxx --arg2 yyy`",
    )
    return ap.parse_args()


def build_train_command(args: argparse.Namespace) -> List[str]:
    """构建对正式训练入口的调用命令。

    输入：
    - args: smoke 参数对象。

    输出：
    - List[str]：可直接传给 `subprocess.run` 的命令列表。
    """
    repo_root = Path(__file__).resolve().parents[1]
    run_train_path = repo_root / "train" / "run_train.py"

    cmd: List[str] = [
        sys.executable,
        str(run_train_path),
        "--prompts_csv",
        str(args.prompts_csv),
        "--output_root",
        str(args.output_root),
        "--experiment_preset",
        str(args.experiment_preset),
        "--model_id",
        str(args.model_id),
        "--model_local_dir",
        str(args.model_local_dir),
        "--local_files_only" if bool(args.local_files_only) else "--no-local_files_only",
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--steps",
        str(int(args.steps)),
        "--guidance_scale",
        str(float(args.guidance_scale)),
        "--resolution",
        str(int(args.resolution)),
        "--validation_prompts",
        str(int(args.validation_prompts)),
        "--stage2_train_prompts",
        str(int(args.stage2_train_prompts)),
        "--stage1_train_prompts",
        str(int(args.stage1_train_prompts)),
        "--calibration_prompts",
        str(int(args.calibration_prompts)),
        "--num_step_buckets",
        str(int(args.num_step_buckets)),
        "--shard_prompts",
        str(int(args.shard_prompts)),
        "--max_prompts_debug",
        str(int(args.max_prompts_debug)),
        "--expansion_factor",
        str(int(args.expansion_factor)),
        "--top_k",
        str(int(args.top_k)),
        "--auxk",
        str(int(args.auxk)),
        "--dead_tokens_threshold",
        str(int(args.dead_tokens_threshold)),
        "--use_block_in_adapter" if bool(args.use_block_in_adapter) else "--no-use_block_in_adapter",
        "--align_weight_target",
        str(float(args.align_weight_target)),
        "--align_warmup_ratio",
        str(float(args.align_warmup_ratio)),
        "--tokens_per_step_target",
        str(int(args.tokens_per_step_target)),
        "--group_bs_stage1",
        str(int(args.group_bs_stage1)),
        "--group_bs_stage2",
        str(int(args.group_bs_stage2)),
        "--run_stage1" if bool(args.run_stage1) else "--no-run_stage1",
        "--run_stage3" if bool(args.run_stage3) else "--no-run_stage3",
        "--run_stage4" if bool(args.run_stage4) else "--no-run_stage4",
        "--use_time_branch" if bool(args.use_time_branch) else "--no-use_time_branch",
        "--time_branch_mode",
        str(args.time_branch_mode),
        "--use_spatial_branch" if bool(args.use_spatial_branch) else "--no-use_spatial_branch",
        "--spatial_branch_mode",
        str(args.spatial_branch_mode),
        "--save_every_steps",
        str(int(args.save_every_steps)),
        "--log_every_steps",
        str(int(args.log_every_steps)),
    ]
    if args.extra_args:
        cmd.extend(list(args.extra_args))
    return cmd


def main() -> None:
    """执行 smoke 训练。

    输入：
    - 无（读取命令行参数）。

    输出：
    - 无；会调用正式训练入口并返回其退出码。
    """
    args = parse_args()
    cmd = build_train_command(args)
    print("[smoke] 即将执行命令：")
    print(" ".join(cmd))
    if bool(args.print_only):
        return
    subprocess.run(cmd, check=True, env=_sanitize_omp_env(os.environ))


if __name__ == "__main__":
    main()
