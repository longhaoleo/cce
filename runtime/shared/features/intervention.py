"""SharedSAE 主线复用的干预辅助函数。"""

from __future__ import annotations

import bisect
import csv
import math
import os
from typing import Dict, List

import torch

from ..io_utils import ensure_dir, safe_name


def _load_coeff_by_step_from_exp53_csv(
    *,
    csv_path: str,
    feature_ids: List[int],
) -> Dict[int, torch.Tensor]:
    """从 exp53 的 feature_time_scores.csv 读取按 step 的系数表。"""
    path = os.path.expanduser(str(csv_path))
    if not path:
        raise ValueError("coeff csv 不能为空。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"coeff csv 不存在: {path}")

    col = "diff"
    feats = [int(x) for x in feature_ids]
    fid_to_pos = {fid: i for i, fid in enumerate(feats)}

    tmp: Dict[int, Dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"step_idx", "feature_id", col}.issubset(fields):
            raise ValueError(f"coeff csv 缺少列: step_idx/feature_id/{col}, got={reader.fieldnames}")
        for row in reader:
            try:
                step = int(row["step_idx"])
                fid = int(row["feature_id"])
                if fid not in fid_to_pos:
                    continue
                val = float(row[col])
            except Exception:
                continue
            tmp.setdefault(step, {})[fid] = val

    out: Dict[int, torch.Tensor] = {}
    for step, fid_map in tmp.items():
        vec = torch.zeros(len(feats), dtype=torch.float32)
        for fid, val in fid_map.items():
            vec[fid_to_pos[int(fid)]] = float(val)
        out[int(step)] = vec
    return out


def _interpolate_coeff_by_step(
    *,
    coeff_by_step: Dict[int, torch.Tensor],
    total_steps: int,
) -> Dict[int, torch.Tensor]:
    """把稀疏 step 系数插值为完整 0..total_steps-1。"""
    if not coeff_by_step:
        return {}
    if total_steps <= 0:
        return dict(coeff_by_step)

    keys = sorted(int(k) for k in coeff_by_step.keys())
    vecs = {int(k): coeff_by_step[int(k)].detach().float().cpu() for k in keys}
    out: Dict[int, torch.Tensor] = {}
    if len(keys) == 1:
        v = vecs[keys[0]]
        for s in range(int(total_steps)):
            out[s] = v.clone()
        return out

    for s in range(int(total_steps)):
        if s in vecs:
            out[s] = vecs[s].clone()
            continue
        idx = bisect.bisect_left(keys, s)
        if idx <= 0:
            out[s] = vecs[keys[0]].clone()
            continue
        if idx >= len(keys):
            out[s] = vecs[keys[-1]].clone()
            continue
        k0 = keys[idx - 1]
        k1 = keys[idx]
        v0 = vecs[k0]
        v1 = vecs[k1]
        if k1 == k0:
            out[s] = v0.clone()
            continue
        alpha = float(s - k0) / float(k1 - k0)
        out[s] = (1.0 - alpha) * v0 + alpha * v1
    return out


def _mean_abs_coeff_strength(coeff_by_step: Dict[int, torch.Tensor]) -> float:
    """估计一个 block 的时间权重强度。"""
    if not coeff_by_step:
        return 1.0
    vals = []
    for v in coeff_by_step.values():
        try:
            vals.append(float(v.detach().abs().mean().item()))
        except Exception:
            continue
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def _build_block_scale_map(
    *,
    blocks: List[str],
    base_scale: float,
    coeffs_by_block: Dict[str, Dict[int, torch.Tensor]],
) -> Dict[str, float]:
    """多 block 自动平衡。"""
    if not blocks:
        return {}
    if len(blocks) == 1:
        return {blocks[0]: float(base_scale)}

    strengths: Dict[str, float] = {}
    for block in blocks:
        strengths[block] = _mean_abs_coeff_strength(coeffs_by_block.get(block, {}))
    target = float(sum(strengths.values()) / len(strengths))
    multi_gain = 1.0 / math.sqrt(float(len(blocks)))

    out: Dict[str, float] = {}
    for block in blocks:
        strength = float(strengths[block])
        bal = target / (strength + 1e-12)
        bal = float(max(0.5, min(2.0, bal)))
        out[block] = float(base_scale) * multi_gain * bal
    print(f"[shared-erase] 多 block 自动平衡: base_scale={base_scale}, multi_gain={multi_gain:.4f}")
    for block in blocks:
        print(f"[shared-erase]   block={block} strength={strengths[block]:.6f} scale={out[block]:.6f}")
    return out


def _save_hook_debug_csv(
    *,
    hooks: Dict[str, object],
    out_dir: str,
    tag: str,
) -> None:
    """导出 hook 每步诊断信息。"""
    ensure_dir(out_dir)
    for block, hk in hooks.items():
        rows = getattr(hk, "debug_rows", None)
        if not rows:
            continue
        out_csv = os.path.join(out_dir, f"diag_{tag}_{safe_name(block)}.csv")
        fields = [
            "step_idx",
            "timestep",
            "active",
            "mode",
            "scale",
            "mean_abs_c_base",
            "mean_abs_w_time",
            "mean_abs_c_final",
            "mean_abs_recon_pre_spatial",
            "mean_abs_recon_final",
            "mean_abs_delta_x",
            "delta_over_x",
            "active_feature_ids_time",
            "active_feature_ids_final",
            "top_feature_ids_final",
            "top_feature_scores_final",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})
        print(f"[shared-erase] 诊断 CSV: {out_csv}")
