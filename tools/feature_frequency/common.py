"""Shared 高频特征统计的公共函数。"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Sequence

import torch

from train.prompt_data import PromptRecord, load_prompts_from_csv


def load_prompt_records(path: str, *, base_seed: int, max_prompts: int) -> List[PromptRecord]:
    """读取 csv/txt prompt 集合。"""
    p = Path(os.path.expanduser(str(path))).resolve()
    if not p.exists():
        raise FileNotFoundError(f"prompts_path 不存在: {p}")

    records: List[PromptRecord]
    if p.suffix.lower() == ".csv":
        records = load_prompts_from_csv(str(p), base_seed=int(base_seed))
    elif p.suffix.lower() == ".txt":
        records = []
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                prompt = str(line).strip()
                if not prompt:
                    continue
                records.append(PromptRecord(prompt_id=len(records), prompt=prompt, seed=int(base_seed) + int(idx)))
    else:
        raise ValueError(f"仅支持 .csv / .txt prompt 文件，当前为: {p.suffix}")

    if int(max_prompts) > 0:
        records = records[: int(max_prompts)]
    if not records:
        raise ValueError(f"读取后 prompt 为空: {p}")
    return records


def dataset_tag(prompts_path: str) -> str:
    """给输出目录构造稳定的 prompt 数据集标签。"""
    path = Path(os.path.expanduser(str(prompts_path))).resolve()
    name = path.stem.strip() or "prompts"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def write_rank_csv(
    *,
    path: Path,
    sorted_ids: List[int],
    active_ratio: torch.Tensor,
    mean_act: torch.Tensor,
    std_act: torch.Tensor,
    blacklist_ids: Sequence[int] | None = None,
) -> None:
    """写全量排序表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    bl_set = {int(x) for x in (blacklist_ids or [])}
    fieldnames = [
        "rank",
        "feature_id",
        "frequency_score",
        "active_ratio",
        "mean_activation",
        "std_activation",
    ]
    include_blacklist = blacklist_ids is not None
    if include_blacklist:
        fieldnames.append("blacklisted")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, fid in enumerate(sorted_ids, start=1):
            idx = int(fid)
            row = {
                "rank": int(rank),
                "feature_id": idx,
                "frequency_score": float(active_ratio[idx].item()),
                "active_ratio": float(active_ratio[idx].item()),
                "mean_activation": float(mean_act[idx].item()),
                "std_activation": float(std_act[idx].item()),
            }
            if include_blacklist:
                row["blacklisted"] = int(idx in bl_set)
            writer.writerow(row)


def write_top_csv(
    *,
    path: Path,
    top_ids: Sequence[int],
    active_ratio: torch.Tensor,
    mean_act: torch.Tensor,
    std_act: torch.Tensor,
) -> None:
    """写 top-k 高频特征表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "feature_id", "rank_by", "rank_score", "active_ratio", "mean_activation", "std_activation"],
        )
        writer.writeheader()
        for rank, fid in enumerate(top_ids, start=1):
            idx = int(fid)
            writer.writerow(
                {
                    "rank": int(rank),
                    "feature_id": idx,
                    "rank_by": "active_ratio",
                    "rank_score": float(active_ratio[idx].item()),
                    "active_ratio": float(active_ratio[idx].item()),
                    "mean_activation": float(mean_act[idx].item()),
                    "std_activation": float(std_act[idx].item()),
                }
            )


def write_blacklist(path: Path, *, blacklist_ids: Sequence[int], quantile: float, threshold: float) -> None:
    """写高频特征黑名单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# shared feature blacklist by prompt-conditioned frequency\n")
        f.write(f"# quantile={float(quantile)}, threshold={float(threshold):.8g}\n")
        for fid in blacklist_ids:
            f.write(f"{int(fid)}\n")


def build_blacklist_ids(
    *,
    sorted_ids: Sequence[int],
    active_ratio: torch.Tensor,
    mean_act: torch.Tensor,
    quantile: float,
    active_ratio_min: float,
    mean_min: float,
    max_features: int,
) -> tuple[list[int], float]:
    """按规则生成 blacklist。"""
    positive = active_ratio[active_ratio > 0]
    if int(positive.numel()) <= 0:
        return [], float("inf")

    q = max(0.0, min(1.0, float(quantile)))
    q_val = float(torch.quantile(positive, q).item())
    threshold = max(float(active_ratio_min), q_val)

    blacklist_ids: List[int] = []
    for fid in sorted_ids:
        idx = int(fid)
        if float(active_ratio[idx].item()) < threshold:
            continue
        if float(mean_act[idx].item()) < float(mean_min):
            continue
        blacklist_ids.append(idx)

    if int(max_features) > 0:
        blacklist_ids = blacklist_ids[: int(max_features)]
    return blacklist_ids, threshold

