"""
损失函数计算。
"""

from __future__ import annotations

from dataclasses import dataclass
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
    decoder_decorr: torch.Tensor


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


def decoder_decorrelation_loss(decoder_weight: torch.Tensor) -> torch.Tensor:
    """计算 decoder 字典列向量的去相关损失。

    输入：
    - decoder_weight: Tensor[d_model, n_dirs]，decoder 权重矩阵。

    输出：
    - Tensor[]：标量损失，仅惩罚 Gram 矩阵的非对角项。

    设计意图：
    - unit norm 只能稳定每个方向自己的尺度；
    - 这里进一步压制不同 feature 方向之间的相关性，
      减少“多个 feature 本质上表达同一件事”的情况。
    """
    dirs = decoder_weight / decoder_weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
    gram = dirs.t() @ dirs
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    offdiag = gram - eye
    return torch.mean(offdiag.pow(2))


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
    decoder_decorr: torch.Tensor,
    auxk_coef: float,
    align_weight: float,
    decoder_decorr_weight: float,
) -> LossBreakdown:
    """组合总损失并返回拆解项。

    输入：
    - recon: 重建损失。
    - auxk: AuxK 损失。
    - align: 对齐损失。
    - decoder_decorr: decoder 去相关损失。
    - auxk_coef: AuxK 权重。
    - align_weight: 对齐权重。
    - decoder_decorr_weight: decoder 去相关权重。

    输出：
    - LossBreakdown：包含 total/recon/auxk/align。
    """
    total = recon + float(auxk_coef) * auxk + float(align_weight) * align + float(decoder_decorr_weight) * decoder_decorr
    return LossBreakdown(total=total, recon=recon, auxk=auxk, align=align, decoder_decorr=decoder_decorr)


def group_forward_losses(
    *,
    forward_cache_by_block: Dict[str, ForwardCache],
    x_norm_by_block: Dict[str, torch.Tensor],
    blocks: list[str] | tuple[str, ...],
    mid_block: str,
    auxk_coef: float,
    align_weight: float,
    decoder_decorr_weight: float = 0.0,
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
    return compose_total_loss(
        recon=recon,
        auxk=aux,
        align=align,
        decoder_decorr=recon.new_tensor(0.0),
        auxk_coef=auxk_coef,
        align_weight=align_weight,
        decoder_decorr_weight=decoder_decorr_weight,
    )
