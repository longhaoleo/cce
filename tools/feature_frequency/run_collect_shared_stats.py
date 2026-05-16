#!/usr/bin/env python3
"""第一遍：收集 Shared prompt-conditioned 高频特征基础统计。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm


def _bootstrap_path() -> None:
    """保证从任意工作目录运行时能导入仓库内模块。"""
    repo_root = Path(__file__).resolve().parents[2]
    for path in (str(repo_root),):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_path()

from runtime.shared.pipeline import (  # noqa: E402
    add_checkpoint_args,
    add_generation_override_args,
    add_model_args,
    load_hooked_pipeline,
    load_shared_checkpoint_bundle,
    resolve_blocks,
    resolve_checkpoint_dir,
    resolve_device_dtype,
    resolve_dtype,
    resolve_generation_hparams,
    resolve_norm_scale_by_block,
)
from runtime.shared.locator import _prompt_activation_mats  # noqa: E402
from runtime.shared.features.scoring import _select_step_indices  # noqa: E402
from runtime.shared.io_utils import block_short_name, ensure_dir  # noqa: E402
from runtime.shared.sae_layout import maybe_use_sae_layout  # noqa: E402
from tools.feature_frequency.common import dataset_tag, load_prompt_records, write_rank_csv, write_top_csv  # noqa: E402


LOG_PREFIX = "shared-freq-pass1"


def build_parser() -> argparse.ArgumentParser:
    """构建基础统计参数。"""
    parser = argparse.ArgumentParser(description="Collect prompt-conditioned Shared feature-frequency statistics")
    g_data = parser.add_argument_group("Prompt 数据")
    g_ckpt = parser.add_argument_group("SharedSAE checkpoint")
    g_model = parser.add_argument_group("SDXL")
    g_run = parser.add_argument_group("采样")
    g_score = parser.add_argument_group("统计参数")

    g_data.add_argument("--prompts_path", type=str, required=True, help="prompt 集合路径（.csv 或 .txt）。")
    g_data.add_argument("--blocks", nargs="+", type=str, default=None, help="要统计的 block 列表；默认继承 checkpoint 配置。")
    g_data.add_argument("--max_prompts", type=int, default=1000, help="最多采样多少条 prompt。")
    g_data.add_argument("--base_seed", type=int, default=42, help="当 txt 行没有 seed 时使用 base_seed + idx。")

    add_checkpoint_args(g_ckpt)
    add_model_args(g_model)
    add_generation_override_args(g_run, prompt_required=False)

    g_score.add_argument("--taris_t_start", type=int, default=1000, help="时间窗口上界（高噪声侧）。")
    g_score.add_argument("--taris_t_end", type=int, default=0, help="时间窗口下界（低噪声侧）。")
    g_score.add_argument("--taris_num_steps", type=int, default=50, help="在窗口内均匀采样多少个 step。")
    g_score.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"], help="每条 prompt 内如何聚合多时间步特征。")
    g_score.add_argument("--feature_top_k", type=int, default=200, help="额外导出 top-k 高频特征。")
    g_score.add_argument("--activation_eps", type=float, default=1e-6, help="判定 prompt 激活该特征的阈值 eps。")
    g_score.add_argument("--output_dir", type=str, default="feature_frequency", help="基础统计输出根目录。")
    g_score.add_argument("--run_name", type=str, default="", help="显式指定本次统计目录名；为空时自动生成。")
    g_score.add_argument("--sae_root", type=str, default="", help="统一 SAE 产物根目录；传入后自动映射到 feature-freq。")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt_dir = resolve_checkpoint_dir(ckpt_dir=str(args.ckpt_dir), output_root=str(args.output_root))
    device, dtype = resolve_device_dtype(str(args.device), resolve_dtype(str(args.dtype)), log_prefix=LOG_PREFIX)
    bundle = load_shared_checkpoint_bundle(ckpt_dir=ckpt_dir, device=device, dtype=dtype)
    blocks = resolve_blocks(requested_blocks=args.blocks, ckpt_cfg=bundle.config)
    norm_scale_by_block = resolve_norm_scale_by_block(
        bundle=bundle,
        blocks=blocks,
        log_prefix=LOG_PREFIX,
        warn_if_missing=False,
    )
    steps, guidance_scale, resolution = resolve_generation_hparams(args=args, ckpt_cfg=bundle.config)
    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix=LOG_PREFIX,
    )

    prompt_records = load_prompt_records(
        str(args.prompts_path),
        base_seed=int(args.base_seed),
        max_prompts=int(args.max_prompts),
    )
    print(f"[{LOG_PREFIX}] checkpoint={ckpt_dir}")
    print(f"[{LOG_PREFIX}] blocks={blocks}")
    print(f"[{LOG_PREFIX}] prompts={len(prompt_records)} steps={steps} guidance={guidance_scale} resolution={resolution}")

    n_features = int(bundle.model.n_dirs)
    sum_feat = {str(block): torch.zeros(n_features, dtype=torch.float64) for block in blocks}
    sumsq_feat = {str(block): torch.zeros(n_features, dtype=torch.float64) for block in blocks}
    active_cnt = {str(block): torch.zeros(n_features, dtype=torch.float64) for block in blocks}
    valid = 0
    failed = 0
    eps = float(args.activation_eps)
    step_indices: List[int] | None = None
    timesteps_ref: List[int] | None = None

    for record in tqdm(prompt_records, desc="shared-freq-pass1"):
        try:
            mats_by_block, timesteps = _prompt_activation_mats(
                pipe=pipe,
                model=bundle.model,
                blocks=blocks,
                norm_scale_by_block=norm_scale_by_block,
                prompt=str(record.prompt),
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                resolution=int(resolution),
                seed=int(record.seed),
            )
            if step_indices is None:
                step_indices = _select_step_indices(
                    timesteps,
                    t_start=int(args.taris_t_start),
                    t_end=int(args.taris_t_end),
                    num_t_samples=int(args.taris_num_steps),
                )
                timesteps_ref = list(timesteps)
            elif list(timesteps) != list(timesteps_ref):
                raise ValueError("不同 prompt 的 scheduler timesteps 不一致。")

            for block in blocks:
                mat = mats_by_block[str(block)][step_indices]
                feat_prompt = mat.mean(dim=0) if str(args.aggregate) == "mean" else mat.max(dim=0).values
                feat64 = feat_prompt.to(dtype=torch.float64)
                sum_feat[str(block)] += feat64
                sumsq_feat[str(block)] += feat64 * feat64
                active_cnt[str(block)] += (feat_prompt > eps).to(dtype=torch.float64)
            valid += 1
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"[{LOG_PREFIX}] 跳过 prompt_id={record.prompt_id} | err={exc}")
            continue

    if valid <= 0:
        raise RuntimeError("没有得到任何有效 prompt，无法统计高频特征。")
    if step_indices is None or timesteps_ref is None:
        raise RuntimeError("未成功初始化时间窗。")

    auto_name = (
        f"shared_prompt_freq_{dataset_tag(str(args.prompts_path))}_"
        f"t{int(min(timesteps_ref[idx] for idx in step_indices))}-{int(max(timesteps_ref[idx] for idx in step_indices))}_"
        f"n{len(step_indices)}"
    )
    run_name = str(args.run_name).strip() or auto_name
    output_dir = maybe_use_sae_layout(
        path_value=str(args.output_dir),
        sae_root=str(getattr(args, "sae_root", "")),
        legacy_default="feature_frequency",
        kind="feature_freq",
    )
    output_root = Path(str(output_dir)).expanduser().resolve() / run_name
    ensure_dir(str(output_root))

    for block in blocks:
        block_key = str(block)
        block_tag = block_short_name(block_key)
        block_out = output_root / block_tag
        ensure_dir(str(block_out))

        mean_act = (sum_feat[block_key] / float(valid)).float()
        var_act = (sumsq_feat[block_key] / float(valid) - (sum_feat[block_key] / float(valid)) ** 2).clamp(min=0.0).float()
        std_act = torch.sqrt(var_act)
        active_ratio = (active_cnt[block_key] / float(valid)).float()

        all_ids = list(range(n_features))
        prim_np = active_ratio.cpu().numpy()
        sec_np = mean_act.cpu().numpy()
        order = np.lexsort((-sec_np, -prim_np)).tolist()
        sorted_ids = [all_ids[int(idx)] for idx in order]
        top_k = int(args.feature_top_k)
        top_ids = sorted_ids[:top_k] if top_k > 0 else sorted_ids

        with (block_out / "dataset_feature_stats_all.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature_id", "active_ratio", "mean_activation", "std_activation"],
            )
            writer.writeheader()
            for fid in range(n_features):
                writer.writerow(
                    {
                        "feature_id": int(fid),
                        "active_ratio": float(active_ratio[fid].item()),
                        "mean_activation": float(mean_act[fid].item()),
                        "std_activation": float(std_act[fid].item()),
                    }
                )

        write_rank_csv(
            path=block_out / "all_feature_frequency_ranked.csv",
            sorted_ids=sorted_ids,
            active_ratio=active_ratio,
            mean_act=mean_act,
            std_act=std_act,
            blacklist_ids=None,
        )
        write_top_csv(
            path=block_out / "top_feature_frequency.csv",
            top_ids=top_ids,
            active_ratio=active_ratio,
            mean_act=mean_act,
            std_act=std_act,
        )
        torch.save(
            {
                "active_ratio": active_ratio,
                "mean_activation": mean_act,
                "std_activation": std_act,
                "sorted_feature_ids": torch.tensor(sorted_ids, dtype=torch.long),
                "selected_step_indices": torch.tensor(step_indices, dtype=torch.long),
                "selected_timesteps": torch.tensor([timesteps_ref[idx] for idx in step_indices], dtype=torch.long),
                "aggregate": str(args.aggregate),
                "block": block_key,
                "shared_sae_checkpoint": str(ckpt_dir),
                "prompts_path": str(Path(str(args.prompts_path)).expanduser().resolve()),
                "feature_activation_eps": eps,
                "num_prompts_used": int(valid),
                "num_prompts_failed": int(failed),
                "steps": int(steps),
                "guidance_scale": float(guidance_scale),
                "resolution": int(resolution),
            },
            block_out / "dataset_feature_stats.pt",
        )
        with (block_out / "run_meta.txt").open("w", encoding="utf-8") as f:
            f.write("mode=shared_prompt_conditioned_feature_stats\n")
            f.write(f"shared_sae_checkpoint={ckpt_dir}\n")
            f.write(f"prompts_path={Path(str(args.prompts_path)).expanduser().resolve()}\n")
            f.write(f"block={block_key}\n")
            f.write(f"num_prompts_requested={int(args.max_prompts)}\n")
            f.write(f"num_prompts_used={int(valid)}\n")
            f.write(f"num_prompts_failed={int(failed)}\n")
            f.write(f"steps={int(steps)}\n")
            f.write(f"guidance_scale={float(guidance_scale)}\n")
            f.write(f"resolution={int(resolution)}\n")
            f.write(f"selected_step_indices={' '.join(str(int(x)) for x in step_indices)}\n")
            f.write(f"selected_timesteps={' '.join(str(int(timesteps_ref[idx])) for idx in step_indices)}\n")
            f.write(f"aggregate={str(args.aggregate)}\n")
            f.write(f"feature_top_k={int(args.feature_top_k)}\n")
            f.write(f"feature_activation_eps={eps}\n")

        print(f"[{LOG_PREFIX}] block={block_key} used={valid} failed={failed} stats={block_out}")

    print(f"[{LOG_PREFIX}] run_dir={output_root}")


if __name__ == "__main__":
    main()
