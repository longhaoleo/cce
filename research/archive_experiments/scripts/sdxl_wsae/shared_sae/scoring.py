"""SharedSAE 主线复用的概念打分与导出工具。"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from sdxl_wsae.utils import ensure_dir


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_blacklist_ids(path: str) -> set[int]:
    """读取黑名单 id（支持 txt/csv）。"""
    ids: set[int] = set()
    p = os.path.expanduser(str(path))
    if not os.path.exists(p):
        return ids
    try:
        if p.endswith(".csv"):
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "feature_id" in reader.fieldnames:
                    for row in reader:
                        try:
                            ids.add(int(row["feature_id"]))
                        except Exception:
                            continue
                    return ids
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                s = s.split(",")[0].strip()
                try:
                    ids.add(int(s))
                except Exception:
                    continue
    except Exception:
        return ids
    return ids


def _load_concept_prompts_from_json(
    *,
    concept_name: str,
    concept_root: str = "./target_concept_dict",
) -> Tuple[List[str], List[str], Dict[str, Any], Path]:
    """从 `concept_root/{concept_name}.json` 读取正/负 prompts。"""
    name = str(concept_name).strip()
    if not name:
        raise ValueError("需要提供 concept_name。")

    root_raw = str(concept_root or "./target_concept_dict").strip()
    root = Path(os.path.expanduser(root_raw))
    candidates = [root]
    if not root.is_absolute():
        candidates = [(Path.cwd() / root).resolve(), (REPO_ROOT / root).resolve()]

    path = None
    for candidate_root in candidates:
        candidate = (candidate_root / f"{name}.json").resolve()
        if candidate.exists():
            path = candidate
            break
    if path is None:
        tried = " | ".join(str((candidate_root / f"{name}.json").resolve()) for candidate_root in candidates)
        raise FileNotFoundError(f"未找到概念 json: tried={tried}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"概念 json 必须是 dict: {path}")

    def _as_list(x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            s = x.strip()
            return [s] if s else []
        if isinstance(x, list):
            out: List[str] = []
            for item in x:
                if item is None:
                    continue
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
                    continue
                if isinstance(item, dict):
                    for key in ("prompt", "text", "caption", "value"):
                        if key in item:
                            out.extend(_as_list(item.get(key)))
                            break
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        if isinstance(x, dict):
            for key in ("prompts", "prompt", "texts", "text", "captions"):
                if key in x:
                    return _as_list(x.get(key))
            out: List[str] = []
            for value in x.values():
                out.extend(_as_list(value))
            return out
        raise ValueError(f"prompts 字段必须是 string 或 list: got {type(x)}")

    pos_keys = (
        "pos_prompts",
        "pos",
        "positive_prompts",
        "positive",
        "concept_prompts",
        "concept",
        "target_prompts",
        "target",
    )
    neg_keys = (
        "neg_prompts",
        "neg",
        "negative_prompts",
        "negative",
        "non_concept_prompts",
        "non_concept",
        "control_prompts",
        "control",
        "anti_prompts",
        "anti",
    )

    def _pick_first(keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key in data:
                return data.get(key)
        return None

    pos = _as_list(_pick_first(pos_keys))
    neg = _as_list(_pick_first(neg_keys))
    return pos, neg, data, path


def _select_step_indices(
    timesteps: Sequence[int],
    *,
    t_start: int,
    t_end: int,
    num_t_samples: int,
) -> List[int]:
    """从 scheduler timesteps 中选出窗口内的 step 索引，并在窗口内均匀采样。"""
    low = int(min(t_start, t_end))
    high = int(max(t_start, t_end))
    candidates = [i for i, t in enumerate(timesteps) if low <= int(t) <= high]
    if not candidates:
        candidates = list(range(len(timesteps)))

    n = int(num_t_samples)
    if n <= 0 or len(candidates) <= n:
        return candidates

    pick_pos = np.linspace(0, len(candidates) - 1, num=n, dtype=int).tolist()
    picked = [candidates[i] for i in pick_pos]
    out: List[int] = []
    seen = set()
    for idx in picked:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _taris_score(
    *,
    pos_mu: torch.Tensor,
    neg_mu: torch.Tensor,
    step_indices: Sequence[int],
    delta: float,
) -> torch.Tensor:
    """按 TARIS 公式对选定 step 做时间平均。"""
    if pos_mu.shape != neg_mu.shape:
        raise ValueError(f"pos/neg 形状不一致: {tuple(pos_mu.shape)} vs {tuple(neg_mu.shape)}")
    if not step_indices:
        raise ValueError("step_indices 为空，无法计算 TARIS。")

    d = float(delta)
    scores = torch.zeros(int(pos_mu.shape[1]), dtype=torch.float32)
    for idx in step_indices:
        mu_c = pos_mu[int(idx)]
        mu_nc = neg_mu[int(idx)]
        e_c = float(mu_c.sum().item()) + d
        e_nc = float(mu_nc.sum().item()) + d
        scores += (mu_c / e_c) - (mu_nc / e_nc)
    return scores / float(len(step_indices))


def _saeuron_score_v2(
    *,
    pos_mu: torch.Tensor,
    neg_mu: torch.Tensor,
    neg_std: torch.Tensor,
    step_indices: Sequence[int],
    epsilon: float,
) -> torch.Tensor:
    """按 SAeUron 风格公式计算时域平均方差惩罚分数。"""
    if pos_mu.shape != neg_mu.shape:
        raise ValueError(f"pos/neg 形状不一致: {tuple(pos_mu.shape)} vs {tuple(neg_mu.shape)}")
    if pos_mu.shape != neg_std.shape:
        raise ValueError(f"neg_std 形状不一致: {tuple(neg_std.shape)} vs {tuple(pos_mu.shape)}")
    if not step_indices:
        raise ValueError("step_indices 为空，无法计算 SAeUron score。")

    eps = float(epsilon)
    scores = torch.zeros(int(pos_mu.shape[1]), dtype=torch.float32)
    for idx in step_indices:
        mu_pos = pos_mu[int(idx)]
        mu_neg = neg_mu[int(idx)]
        sigma_neg = neg_std[int(idx)]
        scores += (mu_pos - mu_neg) / (sigma_neg + eps)
    return scores / float(len(step_indices))


def _compute_desc_ranks(scores: torch.Tensor) -> torch.Tensor:
    """按分数降序计算 rank（1 = 最佳）。"""
    n = int(scores.numel())
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty(n, dtype=torch.long)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.long)
    return ranks


def _save_score_compare_csv(
    path: str,
    *,
    taris_scores: torch.Tensor,
    saeuron_scores: torch.Tensor,
) -> None:
    """保存 TARIS 与 SAeUron 分数对比（含排名变化）。"""
    ensure_dir(os.path.dirname(path) or ".")
    taris = taris_scores.detach().float().cpu()
    saeuron = saeuron_scores.detach().float().cpu()
    if taris.shape != saeuron.shape:
        raise ValueError(f"对比分数形状不一致: {tuple(taris.shape)} vs {tuple(saeuron.shape)}")

    ranks_taris = _compute_desc_ranks(taris)
    ranks_saeuron = _compute_desc_ranks(saeuron)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id",
                "taris_score",
                "saeuron_score",
                "taris_rank",
                "saeuron_rank",
                "rank_delta_saeuron_minus_taris",
            ],
        )
        writer.writeheader()
        for fid in range(int(taris.numel())):
            rt = int(ranks_taris[int(fid)].item())
            rs = int(ranks_saeuron[int(fid)].item())
            writer.writerow(
                {
                    "feature_id": int(fid),
                    "taris_score": float(taris[int(fid)].item()),
                    "saeuron_score": float(saeuron[int(fid)].item()),
                    "taris_rank": rt,
                    "saeuron_rank": rs,
                    "rank_delta_saeuron_minus_taris": int(rs - rt),
                }
            )


def _save_topk_csv(path: str, *, top_ids: torch.Tensor, top_vals: torch.Tensor) -> None:
    """保存 Top-K 特征列表。"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "feature_id", "score"])
        writer.writeheader()
        for rank, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            writer.writerow({"rank": int(rank), "feature_id": int(fid), "score": float(val)})


def _save_feature_time_scores_csv(
    path: str,
    *,
    timesteps: Sequence[int],
    feature_ids: Sequence[int],
    scores_primary: torch.Tensor,
    score_mode: str,
    pos_mu: torch.Tensor,
    neg_mu: torch.Tensor,
    neg_std: torch.Tensor,
    taris_scores: torch.Tensor | None = None,
    saeuron_scores: torch.Tensor | None = None,
    saeuron_eps: float = 1e-6,
    delta: float,
) -> None:
    """保存特征按 step 的激活曲线与归一化差分。"""
    ensure_dir(os.path.dirname(path) or ".")
    ids = [int(x) for x in feature_ids]
    d = float(delta)
    eps = float(saeuron_eps)
    mode = str(score_mode).strip().lower()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "timestep",
                "feature_id",
                "score_mode",
                "primary_score",
                "taris_score",
                "saeuron_score",
                "energy_pos",
                "energy_neg",
                "pos_mu_raw",
                "neg_mu_raw",
                "neg_sigma_raw",
                "diff_raw",
                "saeuron_term_raw",
                "pos_mu_norm",
                "neg_mu_norm",
                "diff",
            ],
        )
        writer.writeheader()
        for step_idx, timestep in enumerate(list(timesteps)):
            e_pos = float(pos_mu[int(step_idx)].sum().item()) + d
            e_neg = float(neg_mu[int(step_idx)].sum().item()) + d
            for fid in ids:
                pos_raw = float(pos_mu[int(step_idx), int(fid)].item())
                neg_raw = float(neg_mu[int(step_idx), int(fid)].item())
                neg_sigma = float(neg_std[int(step_idx), int(fid)].item())
                pos_norm = float(pos_raw / e_pos)
                neg_norm = float(neg_raw / e_neg)
                taris_val = (
                    float(taris_scores[int(fid)].item())
                    if taris_scores is not None
                    else (float(scores_primary[int(fid)].item()) if mode == "taris" else "")
                )
                saeuron_val = (
                    float(saeuron_scores[int(fid)].item())
                    if saeuron_scores is not None
                    else (float(scores_primary[int(fid)].item()) if mode == "saeuron" else "")
                )
                writer.writerow(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(timestep),
                        "feature_id": int(fid),
                        "score_mode": str(mode),
                        "primary_score": float(scores_primary[int(fid)].item()),
                        "taris_score": taris_val,
                        "saeuron_score": saeuron_val,
                        "energy_pos": float(e_pos),
                        "energy_neg": float(e_neg),
                        "pos_mu_raw": float(pos_raw),
                        "neg_mu_raw": float(neg_raw),
                        "neg_sigma_raw": float(neg_sigma),
                        "diff_raw": float(pos_raw - neg_raw),
                        "saeuron_term_raw": float((pos_raw - neg_raw) / (neg_sigma + eps)),
                        "pos_mu_norm": float(pos_norm),
                        "neg_mu_norm": float(neg_norm),
                        "diff": float(pos_norm - neg_norm),
                    }
                )


def _topk_with_blacklist(
    *,
    scores: torch.Tensor,
    top_k: int,
    blacklist_ids: set[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在黑名单过滤后，返回分数的正向/负向 Top-K。"""
    scores_pos = scores.clone()
    scores_neg = scores.clone()
    if blacklist_ids:
        valid = [int(fid) for fid in blacklist_ids if 0 <= int(fid) < int(scores.numel())]
        if valid:
            idx = torch.tensor(valid, dtype=torch.long)
            scores_pos[idx] = -float("inf")
            scores_neg[idx] = float("inf")

    k = max(1, int(top_k))
    top_pos_vals, top_pos_ids = torch.topk(scores_pos, k=min(k, int(scores.numel())))
    top_neg_vals, top_neg_ids = torch.topk(scores_neg, k=min(k, int(scores.numel())), largest=False)
    return top_pos_ids, top_pos_vals, top_neg_ids, top_neg_vals

