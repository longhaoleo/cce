"""
损失函数计算。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch

from SAE import ForwardCache


@dataclass
class LossBreakdown:
    """损失拆解结果。

    输入：
    - 各字段由损失函数计算得到。

    输出：
    - 统一结构对象，便于日志记录。
    """

    total: torch.Tensor
    recon: torch.Tensor
    auxk: torch.Tensor
    align: torch.Tensor
    latent_decorr: torch.Tensor


def recon_loss(x_hat: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    """计算重建 MSE 损失。

    输入：
    - x_hat: Tensor[N,d]，模型重建输出。
    - x_norm: Tensor[N,d]，归一化后的目标输入。

    输出：
    - Tensor[]：标量 MSE。
    """
    return torch.mean((x_hat - x_norm) ** 2)


def auxk_loss(x_aux: Optional[torch.Tensor], x_norm: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """计算 AuxK 残差重建损失。

    输入：
    - x_aux: Tensor[N,d] 或 None，AuxK 分支输出。
    - x_norm: Tensor[N,d]，归一化输入。
    - x_hat: Tensor[N,d]，主分支重建。

    输出：
    - Tensor[]：标量损失；若未启用 AuxK 返回 0。
    """
    if x_aux is None:
        return x_norm.new_tensor(0.0)
    residual = (x_norm - x_hat).detach()
    return torch.mean((x_aux - residual) ** 2)


def align_loss_from_group_latents(
    z_by_block: Dict[str, torch.Tensor],
    *,
    mid_block: str,
    blocks: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """计算以 mid block 为锚点的对齐损失。

    输入：
    - z_by_block: 每个 block 的 token latent，形状 `[N,m]`。
    - mid_block: 锚点 block 名称。
    - blocks: 参与对齐的 block 列表。

    输出：
    - Tensor[]：标量对齐损失。
    """
    if mid_block not in z_by_block:
        return next(iter(z_by_block.values())).new_tensor(0.0)
    q_mid = _pooled_unit(z_by_block[mid_block])
    loss = q_mid.new_tensor(0.0)
    for b in blocks:
        if b == mid_block:
            continue
        if b not in z_by_block:
            continue
        q = _pooled_unit(z_by_block[b])
        # PLAN 里定义的是 ||q_b - q_mid||_2^2，而不是按特征维做均值。
        loss = loss + torch.sum((q - q_mid) ** 2)
    return loss


def latent_covariance_decorrelation_loss(
    z_by_block: Dict[str, torch.Tensor],
    *,
    top_k: int,
    mode: str = "token",
    pool: str = "mean",
    pool_topq: float = 0.1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """对 batch 内最活跃 latent 做协方差去相关。

    这个正则直接惩罚“哪些 feature 在数据上经常一起亮”，比只约束
    decoder 方向更贴近擦除时的副作用问题。
    """
    if not z_by_block:
        raise ValueError("z_by_block 不能为空")
    first = next(iter(z_by_block.values()))
    k = int(top_k)
    if k <= 0:
        return first.new_tensor(0.0)

    z_items = [z.float() for z in z_by_block.values() if z.numel() > 0]
    if not z_items:
        return first.new_tensor(0.0)
    mode_norm = str(mode).strip().lower()
    pool_norm = str(pool).strip().lower()
    if mode_norm not in {"token", "block_pooled"}:
        raise ValueError(f"未知 latent_decorr_mode: {mode}")
    if pool_norm not in {"mean", "topq", "hybrid"}:
        raise ValueError(f"未知 latent_decorr_pool: {pool}")
    q = float(pool_topq)
    if not (0.0 < q <= 1.0):
        raise ValueError("latent_decorr_pool_topq 必须在 (0, 1] 内")
    e = float(eps)
    if e <= 0.0:
        raise ValueError("latent_decorr_eps 必须 > 0")

    if mode_norm == "token":
        z = torch.cat(z_items, dim=0)
    else:
        z = torch.cat([_pool_block_latent(z, pool=pool_norm, topq=q) for z in z_items], dim=0)
    if int(z.shape[0]) < 2 or int(z.shape[1]) < 2:
        return z.new_tensor(0.0)

    # 只看 batch 内 top-active feature，避免构造完整 n_dirs x n_dirs Gram。
    activity = z.detach().mean(dim=0)
    top_n = min(k, int(activity.numel()))
    vals, idx = torch.topk(activity, k=top_n)
    idx = idx[vals > 0]
    if int(idx.numel()) < 2:
        return z.new_tensor(0.0)

    z_sel = z[:, idx]
    z_sel = z_sel - z_sel.mean(dim=0, keepdim=True)
    std = z_sel.pow(2).mean(dim=0).sqrt()
    valid = std > e
    if int(valid.sum().item()) < 2:
        return z.new_tensor(0.0)

    z_norm = z_sel[:, valid] / std[valid].clamp_min(e).unsqueeze(0)
    corr = z_norm.t() @ z_norm / float(max(1, int(z_norm.shape[0]) - 1))
    corr = corr - torch.diag_embed(torch.diagonal(corr))
    return torch.mean(corr.pow(2))


def _pool_block_latent(z: torch.Tensor, *, pool: str, topq: float) -> torch.Tensor:
    """把单个 block 的 token latent 池化成 `[1, n_features]`。"""
    if int(z.shape[0]) <= 0:
        return z.new_zeros((1, int(z.shape[1])))
    pool_norm = str(pool).strip().lower()
    mean = z.mean(dim=0, keepdim=True)
    if pool_norm == "mean":
        return mean

    k = max(1, int(math.ceil(float(z.shape[0]) * float(topq))))
    k = min(k, int(z.shape[0]))
    topq_mean = torch.topk(z, k=k, dim=0).values.mean(dim=0, keepdim=True)
    if pool_norm == "topq":
        return topq_mean
    if pool_norm == "hybrid":
        return 0.5 * mean + 0.5 * topq_mean
    raise ValueError(f"未知 latent_decorr_pool: {pool}")


def _pooled_unit(z: torch.Tensor) -> torch.Tensor:
    """对 token latent 做池化并单位化。

    输入：
    - z: Tensor[N,m]。

    输出：
    - Tensor[m]：mean pooling 后 L2 标准化向量。
    """
    pooled = z.mean(dim=0)
    return pooled / pooled.norm(p=2).clamp_min(1e-12)


def compose_total_loss(
    *,
    recon: torch.Tensor,
    auxk: torch.Tensor,
    align: torch.Tensor,
    latent_decorr: torch.Tensor,
    auxk_coef: float,
    align_weight: float,
    latent_decorr_weight: float,
) -> LossBreakdown:
    """组合总损失并返回拆解项。

    输入：
    - recon: 重建损失。
    - auxk: AuxK 损失。
    - align: 对齐损失。
    - auxk_coef: AuxK 权重。
    - align_weight: 对齐权重。
    - latent_decorr_weight: latent 协方差去相关权重。

    输出：
    - LossBreakdown：包含 total/recon/auxk/align。
    """
    total = (
        recon
        + float(auxk_coef) * auxk
        + float(align_weight) * align
        + float(latent_decorr_weight) * latent_decorr
    )
    return LossBreakdown(
        total=total,
        recon=recon,
        auxk=auxk,
        align=align,
        latent_decorr=latent_decorr,
    )


def group_forward_losses(
    *,
    forward_cache_by_block: Dict[str, ForwardCache],
    x_norm_by_block: Dict[str, torch.Tensor],
    blocks: list[str] | tuple[str, ...],
    mid_block: str,
    auxk_coef: float,
    align_weight: float,
    latent_decorr_weight: float = 0.0,
    latent_decorr_top_k: int = 256,
    latent_decorr_mode: str = "token",
    latent_decorr_pool: str = "mean",
    latent_decorr_pool_topq: float = 0.1,
    latent_decorr_eps: float = 1e-4,
) -> LossBreakdown:
    """计算单个 group 的损失。

    输入：
    - forward_cache_by_block: 各 block 的前向输出缓存。
    - x_norm_by_block: 各 block 对应的 `x_norm`。
    - blocks: 当前阶段参与训练的 block 列表。
    - mid_block: 对齐锚点。
    - auxk_coef: AuxK 权重。
    - align_weight: 对齐损失权重。

    输出：
    - LossBreakdown：该 group 的损失拆解结果。
    """
    recon_items = []
    aux_items = []
    z_by_block: Dict[str, torch.Tensor] = {}

    for b in blocks:
        if b not in forward_cache_by_block:
            continue
        cache = forward_cache_by_block[b]
        x_norm = x_norm_by_block[b]
        recon_items.append(recon_loss(cache.x_hat, x_norm))
        aux_items.append(auxk_loss(cache.x_aux, x_norm, cache.x_hat))
        z_by_block[b] = cache.z

    if not recon_items:
        raise RuntimeError("group 内没有可用 block 结果，无法计算损失")

    recon = torch.stack(recon_items).mean()
    aux = torch.stack(aux_items).mean() if aux_items else recon.new_tensor(0.0)
    align = align_loss_from_group_latents(z_by_block, mid_block=mid_block, blocks=blocks)
    latent_decorr = latent_covariance_decorrelation_loss(
        z_by_block,
        top_k=int(latent_decorr_top_k),
        mode=str(latent_decorr_mode),
        pool=str(latent_decorr_pool),
        pool_topq=float(latent_decorr_pool_topq),
        eps=float(latent_decorr_eps),
    )
    return compose_total_loss(
        recon=recon,
        auxk=aux,
        align=align,
        latent_decorr=latent_decorr,
        auxk_coef=auxk_coef,
        align_weight=align_weight,
        latent_decorr_weight=latent_decorr_weight,
    )
