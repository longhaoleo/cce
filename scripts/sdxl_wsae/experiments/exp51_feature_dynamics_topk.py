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
    """Top-K 热图叠加的主流程（对每个 block 各自输出一组图 + 指标）。"""
    ensure_dir(os.path.join(output_dir, f"sae_delta_vis_top{sae_top_k}"))
    projector = SAEFeatureProjector()
    stride = max(1, int(delta_stride))

    for block in blocks:
        deltas = deltas_by_block.get(block, [])
        if not deltas:
            continue
        sae = saes[block]

        per_block_dir = os.path.join(output_dir, f"sae_delta_vis_top{sae_top_k}", safe_name(block))
        ensure_dir(per_block_dir)

        rows = []
        x_idx: List[int] = []
        top1_scores: List[float] = []
        topk_scores: List[float] = []

        for item in deltas:
            heat_2d, top_ids, top_vals, topk_mass = projector.topk_heatmap(
                sae,
                x=item.x,
                hw=item.hw,
                top_k=int(sae_top_k),
            )
            top1 = float(top_vals[0].item())
            top1_id = int(top_ids[0].item())

            rows.append(
                {
                    "block": block,
                    "step_idx": int(item.step_idx),
                    "timestep": int(item.timestep),
                    "top1_feature_id": top1_id,
                    "top1_score": top1,
                    "topk_mass": float(topk_mass),
                    "topk_ids": " ".join(str(int(x)) for x in top_ids.tolist()),
                    "topk_scores": " ".join(f"{float(v):.6f}" for v in top_vals.tolist()),
                }
            )
            x_idx.append(int(item.step_idx))
            top1_scores.append(top1)
            topk_scores.append(float(topk_mass))

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

        if rows:
            csv_path = os.path.join(output_dir, f"metrics_{safe_name(block)}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            curve_path = os.path.join(output_dir, f"update_curve_{safe_name(block)}.png")
            plt.figure(figsize=(10, 4))
            plt.plot(x_idx, top1_scores, marker="o", label="top1_score")
            plt.plot(x_idx, topk_scores, marker="s", label="topk_mass")
            plt.xlabel("step_idx")
            plt.ylabel("activation")
            plt.title(f"SAE Feature Dynamics: {block}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(curve_path, dpi=150)
            plt.close()

            print(f"[{block}] 指标 CSV: {csv_path}")
            print(f"[{block}] 曲线图:   {curve_path}")


def run_exp51_feature_dynamics_topk(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    viz_cfg: VizConfig,
) -> None:
    """实验 51：Top-K 热图叠加。"""
    blocks = list(sae_cfg.blocks)
    session, base_image, _timesteps, deltas_by_block = prepare_deltas_for_blocks(
        model_cfg=model_cfg,
        sae_cfg=sae_cfg,
        run_cfg=run_cfg,
        output_dir=viz_cfg.output_dir,
        blocks=blocks,
    )
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
    print(f"实验 51 完成（topk），输出目录: {viz_cfg.output_dir}")

