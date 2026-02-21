"""
实验 51：特征动力学 Top-K 热图叠加。

输出内容：
- `generated_image.png`：本次采样的最终生成图（由 shared_prepare 保存）
- `sae_delta_vis_top{K}/.../*.png`：每个 block 每隔 stride 的热力图叠加图
- `metrics_{block}.csv`：每步 top1/topk 指标
- `update_curve_{block}.png`：top1_score/topk_mass 随 step 变化曲线

核心思想：
- 对每个 step 的 delta = h_out - h_in，用 SAE.encode 得到稀疏激活；
- 取 top-k 特征，用 decoder 只重建这些特征对应的分量；
- 对每个 token 的重建向量取 L2 范数，reshape 成 (H,W) 得到空间热图；
- 叠加到最终生成图上，观察“更新发生在哪里”。
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from ..configs import RunConfig, SAEConfig, VizConfig
from ..utils import ensure_dir, normalize_01, overlay_heatmap, safe_name
from .shared_prepare import StepDelta, prepare_deltas_for_blocks


class SAEFeatureProjector:
    """把 token-delta 投影为“空间热力图”的工具。"""

    @staticmethod
    @torch.no_grad()
    def topk_heatmap(
        sae: torch.nn.Module,
        *,
        x: torch.Tensor,
        hw: Tuple[int, int],
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        d_model = int(getattr(sae, "d_model"))
        if int(x.shape[-1]) != d_model:
            raise ValueError(f"输入维度不匹配: {x.shape[-1]} != sae.d_model={d_model}")

        p = next(sae.parameters())
        x = x.to(device=p.device, dtype=p.dtype)
        z = sae.encode(x)  # [tokens, n_features]

        scores = z.mean(dim=0)  # [n_features]
        k = max(1, min(int(top_k), int(scores.shape[0])))
        vals, inds = torch.topk(scores, k=k, dim=0)

        # 只用 top-k 特征做重建，并把每个 token 的 L2 范数作为空间强度。
        z_top = z[:, inds]  # [tokens, k]
        w_top = sae.decoder.weight[:, inds].to(device=z_top.device, dtype=z_top.dtype)  # [d_model, k]
        recon = z_top @ w_top.t()  # [tokens, d_model]
        heat = torch.norm(recon, dim=1)  # [tokens]

        h, w = int(hw[0]), int(hw[1])
        if h * w <= 0 or int(heat.shape[0]) % (h * w) != 0:
            raise ValueError(f"hw 与 token 数不匹配: hw={hw}, tokens={heat.shape[0]}")
        b = int(heat.shape[0] // (h * w))
        heat_2d = heat.reshape(b, h, w)[0].detach().float().cpu()
        heat_2d = normalize_01(heat_2d)
        return heat_2d, inds.detach().cpu(), vals.detach().cpu(), float(vals.sum().item())

    @staticmethod
    @torch.no_grad()
    def fixed_features_heatmap(
        sae: torch.nn.Module,
        *,
        x: torch.Tensor,
        hw: Tuple[int, int],
        feature_ids: List[int],
        coeff_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """用“指定特征集合”做重建的空间热图（直接走 SAE.decoder）。

        返回：
        - heat_2d: [H, W]，对指定特征集合重建后的 token L2 强度图
        - used_ids: 实际使用到的 feature ids（过滤越界后）
        - used_vals: 这些特征在该 step 的平均激活（z.mean(dim=0) 对应维度）
        - mass: used_vals.sum()，作为一个简洁的总强度指标
        """
        d_model = int(getattr(sae, "d_model"))
        if int(x.shape[-1]) != d_model:
            raise ValueError(f"输入维度不匹配: {x.shape[-1]} != sae.d_model={d_model}")

        p = next(sae.parameters())
        x = x.to(device=p.device, dtype=p.dtype)
        z = sae.encode(x)  # [tokens, n_features]
        scores = z.mean(dim=0)  # [n_features]

        n_feat = int(scores.shape[0])
        ids = [int(fid) for fid in feature_ids if 0 <= int(fid) < n_feat]
        if not ids:
            raise ValueError("指定 feature_ids 全部越界（或为空），无法可视化。")

        used_vals = scores[ids].detach().float().cpu()
        used_ids = torch.tensor(ids, dtype=torch.long)

        # 你要求的“直接 decoder”：
        # 1) 先对 x 做 SAE.encode 得到稀疏系数 z
        # 2) 只保留指定 feature_ids 的系数，其余置零
        # 3) 直接调用 sae.decoder 重建
        #
        # 注意：这里不加 pre_bias，因为我们在可视化的是“这些特征对应的更新分量”，
        # 而不是要完整重建出 x 本身。
        z_mask = torch.zeros_like(z)
        z_mask[:, ids] = z[:, ids] * float(coeff_scale)
        recon = sae.decoder(z_mask)  # [tokens, d_model]
        heat = torch.norm(recon, dim=1)  # [tokens]

        h, w = int(hw[0]), int(hw[1])
        if h * w <= 0 or int(heat.shape[0]) % (h * w) != 0:
            raise ValueError(f"hw 与 token 数不匹配: hw={hw}, tokens={heat.shape[0]}")
        b = int(heat.shape[0] // (h * w))
        heat_2d = heat.reshape(b, h, w)[0].detach().float().cpu()
        heat_2d = normalize_01(heat_2d)
        return heat_2d, used_ids, used_vals, float(used_vals.sum().item())


def _load_feature_ids_from_csv(path: str, *, k: int) -> List[int]:
    """从 exp53 输出的 csv 读 feature_id 列表。"""
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature csv 不存在: {path}")

    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "feature_id" not in (reader.fieldnames or []):
            raise ValueError(f"csv 缺少 feature_id 列: {path}, columns={reader.fieldnames}")
        for row in reader:
            try:
                ids.append(int(row["feature_id"]))
            except Exception:
                continue
    if not ids:
        raise ValueError(f"csv 未读到任何 feature_id: {path}")
    if int(k) > 0:
        ids = ids[: int(k)]
    return ids


def _run_topk(
    *,
    output_dir: str,
    sae_top_k: int,
    delta_stride: int,
    overlay_alpha: float,
    base_image,
    blocks: List[str],
    deltas_by_block: Dict[str, List[StepDelta]],
    saes: Dict[str, torch.nn.Module],
) -> None:
    """Top-K 热图叠加的主流程（对每个 block 各自输出一组图）。"""
    root = os.path.join(output_dir, f"exp51_topk_k{sae_top_k}")
    ensure_dir(root)
    projector = SAEFeatureProjector()
    stride = max(1, int(delta_stride))

    for block in blocks:
        deltas = deltas_by_block.get(block, [])
        if not deltas:
            continue
        sae = saes[block]

        per_block_dir = os.path.join(root, safe_name(block))
        ensure_dir(per_block_dir)

        for item in deltas:
            heat_2d, top_ids, top_vals, topk_mass = projector.topk_heatmap(
                sae,
                x=item.x,
                hw=item.hw,
                top_k=int(sae_top_k),
            )
            top1 = float(top_vals[0].item())
            top1_id = int(top_ids[0].item())

            if (item.step_idx % stride) == 0:
                out_png = os.path.join(
                    per_block_dir,
                    f"step_{item.step_idx:04d}_t{int(item.timestep)}_top{int(sae_top_k)}_agg.png",
                )
                title = (
                    f"{block}\n"
                    f"step={item.step_idx} t={int(item.timestep)} "
                    f"top1=f{top1_id}({top1:.3f})"
                )
                overlay_heatmap(
                    heat_2d,
                    out_path=out_png,
                    title=title,
                    base_image=base_image,
                    alpha=float(overlay_alpha),
                )


def _run_fixed_features(
    *,
    output_dir: str,
    feature_ids: List[int],
    feature_tag: str,
    delta_stride: int,
    overlay_alpha: float,
    base_image,
    blocks: List[str],
    deltas_by_block: Dict[str, List[StepDelta]],
    saes: Dict[str, torch.nn.Module],
    coeff_scale: float,
) -> None:
    """指定特征集合热图叠加（对每个 block 各自输出一组图）。"""
    root = os.path.join(output_dir, f"exp51_fixed_{feature_tag}")
    ensure_dir(root)
    projector = SAEFeatureProjector()
    stride = max(1, int(delta_stride))

    for block in blocks:
        deltas = deltas_by_block.get(block, [])
        if not deltas:
            continue
        sae = saes[block]

        per_block_dir = os.path.join(root, safe_name(block))
        ensure_dir(per_block_dir)

        for item in deltas:
            heat_2d, used_ids, used_vals, mass = projector.fixed_features_heatmap(
                sae,
                x=item.x,
                hw=item.hw,
                feature_ids=feature_ids,
                coeff_scale=float(coeff_scale),
            )
            top_rel = int(torch.argmax(used_vals).item())
            top1_id = int(used_ids[top_rel].item())
            top1_val = float(used_vals[top_rel].item())

            if (item.step_idx % stride) == 0:
                out_png = os.path.join(
                    per_block_dir,
                    f"step_{item.step_idx:04d}_t{int(item.timestep)}_fixed_{feature_tag}_agg.png",
                )
                title = (
                    f"{block}\n"
                    f"step={item.step_idx} t={int(item.timestep)} "
                    f"fixed={feature_tag} top1=f{top1_id}({top1_val:.3f})"
                )
                overlay_heatmap(
                    heat_2d,
                    out_path=out_png,
                    title=title,
                    base_image=base_image,
                    alpha=float(overlay_alpha),
                )


def run_exp51_feature_dynamics_topk(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    viz_cfg: VizConfig,
) -> None:
    """实验 51：Top-K 热图叠加（或指定特征集合可视化）。"""
    blocks = list(sae_cfg.blocks)
    session, base_image, _timesteps, deltas_by_block = prepare_deltas_for_blocks(
        model_cfg=model_cfg,
        sae_cfg=sae_cfg,
        run_cfg=run_cfg,
        output_dir=viz_cfg.output_dir,
        blocks=blocks,
    )

    fixed_csv = str(getattr(viz_cfg, "exp51_feature_csv", "") or "").strip()
    if fixed_csv:
        feature_ids = _load_feature_ids_from_csv(fixed_csv, k=int(getattr(viz_cfg, "exp51_feature_k", 0)))
        feature_tag = f"k{len(feature_ids)}"
        _run_fixed_features(
            output_dir=viz_cfg.output_dir,
            feature_ids=feature_ids,
            feature_tag=feature_tag,
            delta_stride=viz_cfg.delta_stride,
            overlay_alpha=viz_cfg.overlay_alpha,
            base_image=base_image,
            blocks=blocks,
            deltas_by_block=deltas_by_block,
            saes=session.saes,
            coeff_scale=float(getattr(viz_cfg, "exp51_feature_coeff_scale", 1.0)),
        )
        print(f"实验 51 完成（fixed features），输出目录: {viz_cfg.output_dir}")
        print(f"  fixed_csv: {fixed_csv}")
        print(f"  feature_ids[:10]: {feature_ids[:10]}")
        return

    _run_topk(
        output_dir=viz_cfg.output_dir,
        sae_top_k=viz_cfg.sae_top_k,
        delta_stride=viz_cfg.delta_stride,
        overlay_alpha=viz_cfg.overlay_alpha,
        base_image=base_image,
        blocks=blocks,
        deltas_by_block=deltas_by_block,
        saes=session.saes,
    )
    print(f"实验 51 完成（dynamic topk），输出目录: {viz_cfg.output_dir}")
