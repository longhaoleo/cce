"""
实验 52：特征动力学“瀑布图”（Feature Dynamics Waterfall / Money Plot）。

输出内容（每个 block 一套）：
- `generated_image.png`：本次采样的最终生成图（由 shared_prepare 保存）
- `waterfall_{block}.png`：瀑布图热力图
- `waterfall_{block}.npz`：导出的原始数据包（便于后续定量分析/二次可视化）

核心思想：
- 固定 seed 跑一次完整轨迹；
- 对每个 step 的 delta = h_out - h_in，做 SAE.encode 得到稀疏激活；
- 对 token 维做均值聚合，得到该 step 的激活向量 c_t（长度为 n_features）；
- 将所有 step 堆起来得到矩阵 [steps, n_features]；
- 用“每个特征首次达到峰值的时间”对特征排序，再画热力图。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..configs import RunConfig, SAEConfig, VizConfig
from ..utils import safe_name
from .shared_prepare import StepDelta, prepare_deltas_for_blocks


class FeatureDynamicsWaterfallPlotter:
    """瀑布图计算与绘制。"""

    @staticmethod
    @torch.no_grad()
    def activation_matrix(
        sae: torch.nn.Module,
        deltas: Sequence[StepDelta],
    ) -> Tuple[torch.Tensor, List[int]]:
        if not deltas:
            raise ValueError("没有可用于瀑布图的 delta。")
        p = next(sae.parameters())
        rows: List[torch.Tensor] = []
        timesteps: List[int] = []
        for item in deltas:
            x = item.x.to(device=p.device, dtype=p.dtype)
            z = sae.encode(x)  # [tokens, n_features]
            rows.append(z.mean(dim=0).detach().float().cpu())  # [n_features]
            timesteps.append(int(item.timestep))
        return torch.stack(rows, dim=0), timesteps  # [steps, n_features]

    @staticmethod
    def _normalize_for_plot(matrix: torch.Tensor, mode: str) -> torch.Tensor:
        mode = mode.lower()
        if mode == "none":
            return matrix
        if mode == "global":
            m_min = matrix.min()
            m_max = matrix.max()
            return (matrix - m_min) / (m_max - m_min).clamp_min(1e-8)
        if mode == "row":
            return matrix / matrix.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        raise ValueError(f"不支持的归一化模式: {mode}")

    @staticmethod
    def build_sorted_view(
        c_mat: torch.Tensor,
        *,
        max_features: int,
        norm_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_vals, peak_steps = torch.max(c_mat, dim=0)  # 每个特征的峰值与峰值首次出现 step
        active_ids = torch.nonzero(peak_vals > 0, as_tuple=False).squeeze(1)
        if active_ids.numel() == 0:
            k = min(max(1, max_features if max_features > 0 else c_mat.shape[1]), c_mat.shape[1])
            active_ids = torch.topk(peak_vals, k=k).indices

        if max_features > 0 and active_ids.numel() > max_features:
            keep = torch.topk(peak_vals[active_ids], k=max_features).indices
            active_ids = active_ids[keep]

        feat_peak_steps = peak_steps[active_ids]
        sort_idx = torch.argsort(feat_peak_steps, descending=False)
        sorted_feature_ids = active_ids[sort_idx]

        view = c_mat[:, sorted_feature_ids].T.contiguous()  # [features, steps]
        view = FeatureDynamicsWaterfallPlotter._normalize_for_plot(view, norm_mode)
        return view, sorted_feature_ids, feat_peak_steps[sort_idx]

    @staticmethod
    def save_plot(
        view: torch.Tensor,
        *,
        timesteps: Sequence[int],
        out_path: str,
        title: str,
        cmap: str,
    ) -> None:
        n_features, n_steps = int(view.shape[0]), int(view.shape[1])
        fig_w = max(8.0, min(18.0, 0.25 * n_steps))
        fig_h = max(5.0, min(20.0, 0.012 * n_features + 4.0))
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(view.detach().cpu().numpy(), aspect="auto", cmap=cmap, origin="upper")
        plt.colorbar(label="normalized activation")
        plt.xlabel("denoising step (noise -> image)")
        plt.ylabel("feature index (sorted by first peak step)")
        if timesteps:
            tick_n = min(8, n_steps)
            tick_pos = np.linspace(0, n_steps - 1, num=tick_n, dtype=int)
            tick_labels = [str(int(timesteps[i])) for i in tick_pos.tolist()]
            plt.xticks(tick_pos, tick_labels, rotation=0)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()


def _run_waterfall(
    *,
    output_dir: str,
    blocks: List[str],
    deltas_by_block: Dict[str, List[StepDelta]],
    saes: Dict[str, torch.nn.Module],
    max_features: int,
    norm_mode: str,
    cmap: str,
) -> None:
    plotter = FeatureDynamicsWaterfallPlotter()
    for block in blocks:
        deltas = deltas_by_block.get(block, [])
        if not deltas:
            continue
        sae = saes[block]
        c_mat, timesteps = plotter.activation_matrix(sae, deltas)
        view, sorted_ids, peak_steps = plotter.build_sorted_view(
            c_mat,
            max_features=int(max_features),
            norm_mode=norm_mode,
        )

        block_name = safe_name(block)
        png_path = os.path.join(output_dir, f"waterfall_{block_name}.png")
        npz_path = os.path.join(output_dir, f"waterfall_{block_name}.npz")
        title = (
            f"Feature Dynamics Waterfall\n{block} | "
            f"features={int(view.shape[0])} steps={int(view.shape[1])}"
        )
        plotter.save_plot(
            view,
            timesteps=timesteps,
            out_path=png_path,
            title=title,
            cmap=cmap,
        )
        np.savez_compressed(
            npz_path,
            c_mat=c_mat.numpy(),
            view=view.numpy(),
            timesteps=np.asarray(timesteps, dtype=np.int64),
            sorted_feature_ids=sorted_ids.numpy(),
            peak_steps=peak_steps.numpy(),
        )
        print(f"[{block}] 瀑布图: {png_path}")
        print(f"[{block}] 数据包: {npz_path}")


def run_exp52_feature_dynamics_waterfall(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    viz_cfg: VizConfig,
) -> None:
    """实验 52：瀑布图（Money Plot）。"""
    blocks = list(sae_cfg.blocks)
    session, _base_image, _timesteps, deltas_by_block = prepare_deltas_for_blocks(
        model_cfg=model_cfg,
        sae_cfg=sae_cfg,
        run_cfg=run_cfg,
        output_dir=viz_cfg.output_dir,
        blocks=blocks,
    )
    _run_waterfall(
        output_dir=viz_cfg.output_dir,
        blocks=blocks,
        deltas_by_block=deltas_by_block,
        saes=session.saes,
        max_features=viz_cfg.waterfall_max_features,
        norm_mode=viz_cfg.waterfall_norm,
        cmap=viz_cfg.waterfall_cmap,
    )
    print(f"实验 52 完成（waterfall），输出目录: {viz_cfg.output_dir}")

