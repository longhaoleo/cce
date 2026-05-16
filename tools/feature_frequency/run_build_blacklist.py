#!/usr/bin/env python3
"""第二遍：根据基础统计与筛选规则生成 Shared blacklist。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch


def _bootstrap_path() -> None:
    """保证从任意工作目录运行时能导入仓库内模块。"""
    repo_root = Path(__file__).resolve().parents[2]
    for path in (str(repo_root),):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_path()

from runtime.shared.io_utils import block_short_name, ensure_dir  # noqa: E402
from runtime.shared.sae_layout import maybe_use_sae_layout  # noqa: E402
from tools.feature_frequency.common import build_blacklist_ids, write_blacklist, write_rank_csv, write_top_csv  # noqa: E402


LOG_PREFIX = "shared-freq-pass2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Shared feature blacklists from cached stats")
    parser.add_argument("--stats_dir", type=str, required=True, help="第一遍统计输出的 run 目录。")
    parser.add_argument("--blocks", nargs="+", type=str, default=None, help="可选：只处理这些 block；默认处理 stats_dir 下全部 block。")
    parser.add_argument("--feature_top_k", type=int, default=200, help="同步导出 top-k 高频特征表。")
    parser.add_argument("--blacklist_freq_threshold", type=float, default=0.0, help="active_ratio 正值分布分位数。")
    parser.add_argument("--blacklist_active_ratio_min", type=float, default=0.95, help="最小 active_ratio。")
    parser.add_argument("--blacklist_mean_min", type=float, default=0.0, help="最小 mean_activation。")
    parser.add_argument("--blacklist_max_features", type=int, default=0, help="每层 blacklist 数量上限；0 表示不设上限。")
    parser.add_argument("--concept_dict_freq_root", type=str, default="concept_dict_freq", help="正式 blacklist 输出根目录。")
    parser.add_argument("--concept_dig_freq_root", type=str, default="concept_dict_freq", help="concept-dig-freq 输出根目录。")
    parser.add_argument("--sae_root", type=str, default="", help="统一 SAE 产物根目录；传入后自动映射 blacklist / concept-dig-freq。")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _resolve_block_dirs(stats_dir: Path, requested_blocks: List[str] | None) -> List[Path]:
    """解析要处理的 block stats 目录。"""
    if requested_blocks:
        tags = []
        for block in requested_blocks:
            b = str(block)
            tags.append(block_short_name(b) if "." in b else b)
        out = []
        for tag in tags:
            p = stats_dir / tag
            if not p.exists():
                raise FileNotFoundError(f"stats_dir 下不存在 block 目录: {p}")
            out.append(p)
        return out

    out = []
    for child in sorted(stats_dir.iterdir()):
        if child.is_dir() and (child / "dataset_feature_stats.pt").exists():
            out.append(child)
    if not out:
        raise FileNotFoundError(f"stats_dir 下未发现任何 dataset_feature_stats.pt: {stats_dir}")
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    stats_dir = Path(str(args.stats_dir)).expanduser().resolve()
    if not stats_dir.exists():
        raise FileNotFoundError(f"stats_dir 不存在: {stats_dir}")

    concept_dig_freq_root = Path(
        maybe_use_sae_layout(
            path_value=str(args.concept_dig_freq_root),
            sae_root=str(getattr(args, "sae_root", "")),
            legacy_default="concept_dict_freq",
            kind="concept_dig_freq",
        )
    ).expanduser().resolve()
    blacklist_root = Path(
        maybe_use_sae_layout(
            path_value=str(args.concept_dict_freq_root),
            sae_root=str(getattr(args, "sae_root", "")),
            legacy_default="concept_dict_freq",
            kind="blacklist",
        )
    ).expanduser().resolve()
    ensure_dir(str(concept_dig_freq_root))
    ensure_dir(str(blacklist_root))

    for block_dir in _resolve_block_dirs(stats_dir, args.blocks):
        payload = torch.load(block_dir / "dataset_feature_stats.pt", map_location="cpu", weights_only=False)
        block = str(payload["block"])
        block_tag = block_short_name(block)
        freq_out_dir = concept_dig_freq_root / block_tag
        blacklist_out_dir = blacklist_root / block_tag
        ensure_dir(str(freq_out_dir))
        ensure_dir(str(blacklist_out_dir))

        active_ratio = payload["active_ratio"].float()
        mean_act = payload["mean_activation"].float()
        std_act = payload["std_activation"].float()
        sorted_ids = [int(x) for x in payload["sorted_feature_ids"].tolist()]
        top_k = int(args.feature_top_k)
        top_ids = sorted_ids[:top_k] if top_k > 0 else sorted_ids

        blacklist_ids, threshold = build_blacklist_ids(
            sorted_ids=sorted_ids,
            active_ratio=active_ratio,
            mean_act=mean_act,
            quantile=float(args.blacklist_freq_threshold),
            active_ratio_min=float(args.blacklist_active_ratio_min),
            mean_min=float(args.blacklist_mean_min),
            max_features=int(args.blacklist_max_features),
        )

        write_rank_csv(
            path=freq_out_dir / "all_feature_frequency_ranked.csv",
            sorted_ids=sorted_ids,
            active_ratio=active_ratio,
            mean_act=mean_act,
            std_act=std_act,
            blacklist_ids=blacklist_ids,
        )
        write_top_csv(
            path=freq_out_dir / "top_feature_frequency.csv",
            top_ids=top_ids,
            active_ratio=active_ratio,
            mean_act=mean_act,
            std_act=std_act,
        )
        write_blacklist(
            blacklist_out_dir / "feature_blacklist.txt",
            blacklist_ids=blacklist_ids,
            quantile=float(args.blacklist_freq_threshold),
            threshold=threshold,
        )
        with (blacklist_out_dir / "run_meta.txt").open("w", encoding="utf-8") as f:
            f.write("mode=shared_blacklist_build_from_cached_stats\n")
            f.write(f"stats_dir={stats_dir}\n")
            f.write(f"stats_block_dir={block_dir}\n")
            f.write(f"concept_dig_freq_root={concept_dig_freq_root}\n")
            f.write(f"blacklist_root={blacklist_root}\n")
            f.write(f"block={block}\n")
            f.write(f"shared_sae_checkpoint={payload.get('shared_sae_checkpoint', '')}\n")
            f.write(f"prompts_path={payload.get('prompts_path', '')}\n")
            f.write(f"num_prompts_used={int(payload.get('num_prompts_used', 0))}\n")
            f.write(f"steps={int(payload.get('steps', 0))}\n")
            f.write(f"guidance_scale={float(payload.get('guidance_scale', 0.0))}\n")
            f.write(f"resolution={int(payload.get('resolution', 0))}\n")
            f.write(f"aggregate={payload.get('aggregate', '')}\n")
            f.write(f"feature_top_k={int(args.feature_top_k)}\n")
            f.write(f"blacklist_quantile={float(args.blacklist_freq_threshold)}\n")
            f.write(f"blacklist_active_ratio_min={float(args.blacklist_active_ratio_min)}\n")
            f.write(f"blacklist_mean_min={float(args.blacklist_mean_min)}\n")
            f.write(f"blacklist_max_features={int(args.blacklist_max_features)}\n")
            f.write(f"blacklist_threshold={float(threshold):.8g}\n")
            f.write(f"blacklist_size={len(blacklist_ids)}\n")
            f.write(f"feature_activation_eps={float(payload.get('feature_activation_eps', 0.0))}\n")

        print(
            f"[{LOG_PREFIX}] block={block} blacklist={len(blacklist_ids)} "
            f"threshold={threshold:.6g} -> {blacklist_out_dir / 'feature_blacklist.txt'}"
        )


if __name__ == "__main__":
    main()
