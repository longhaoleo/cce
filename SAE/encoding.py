"""
SharedSAE 时间/空间编码工具函数。

SharedSAE 的时间支路和空间支路都依赖这些基础函数：

- timestep 编码：把扩散步数变成连续输入；
- 坐标编码：把 token 的空间位置变成可学习的几何特征；
- mode / HxW 校验：保证训练和推理沿着同一套几何假设运行。

这些函数故意保持“纯函数”风格，方便训练、推理、统计三边复用。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def build_coords_norm(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """生成归一化的 2D 坐标网格。

    返回形状是 `[h*w, 2]`，两个坐标维度都被映射到 `[-1, 1]`。
    这样后续的空间编码既不依赖绝对分辨率，也能在不同 HxW 下保持可比较的几何尺度。
    """
    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    # 使用 cell center 而不是左上角坐标，能让空间编码对每个 token 更对称。
    u = 2.0 * (grid_r + 0.5) / float(h) - 1.0
    v = 2.0 * (grid_c + 0.5) / float(w) - 1.0
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)


def sincos_1d(values: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """计算 1D 正余弦位置编码。

    输入可以是 `[N]` 或 `[N, 1]`，输出固定为 `[N, embed_dim]`。
    前一半维度是 `sin`，后一半维度是 `cos`。

    这套编码不引入额外参数，适合作为时间支路/空间支路的基础几何表示。
    """
    if int(embed_dim) % 2 != 0:
        raise ValueError("embed_dim 必须为偶数")
    if values.dim() == 2 and values.shape[-1] == 1:
        values = values[:, 0]
    values = values.reshape(-1, 1)
    half = int(embed_dim) // 2
    idx = torch.arange(half, device=values.device, dtype=values.dtype)
    freqs = torch.exp(-math.log(10000.0) * idx / max(1, half - 1))
    angles = values * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def sincos_2d(coords_norm: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """计算 2D 正余弦位置编码。

    做法是把总维度平均拆给 x/y 两个方向：
    - x 使用一半里的 `sin/cos`
    - y 使用另一半里的 `sin/cos`

    这样 2D token 会得到一个既保留行列结构、又无需训练参数的位置表示。
    """
    if int(embed_dim) % 4 != 0:
        raise ValueError("2D 编码的 embed_dim 必须可被 4 整除")
    if coords_norm.dim() != 2 or coords_norm.shape[-1] != 2:
        raise ValueError(f"coords_norm 形状非法: {tuple(coords_norm.shape)}")

    quarter = int(embed_dim) // 4
    idx = torch.arange(quarter, device=coords_norm.device, dtype=coords_norm.dtype)
    freqs = torch.exp(-math.log(10000.0) * idx / max(1, quarter - 1))

    u = coords_norm[:, 0:1]
    v = coords_norm[:, 1:2]
    u_angles = u * freqs.unsqueeze(0)
    v_angles = v * freqs.unsqueeze(0)
    return torch.cat(
        [
            torch.sin(u_angles),
            torch.cos(u_angles),
            torch.sin(v_angles),
            torch.cos(v_angles),
        ],
        dim=-1,
    )


def normalize_timestep(timestep: torch.Tensor | int | float) -> torch.Tensor:
    """将 scheduler timestep 归一化到近似 `[0,1]` 范围。

    当前按 1000 训练步这个 SDXL 常见标度来缩放：
    `t_norm = t / 1000`

    目的不是精确复现 scheduler 内部物理量，而是让时间支路拿到一个量级稳定、
    适合 MLP/线性层处理的连续输入。
    """
    if isinstance(timestep, (int, float)):
        return torch.tensor([float(timestep) / 1000.0], dtype=torch.float32)
    ts = timestep.float().reshape(-1)
    return ts / 1000.0


def ensure_mode(mode: str, *, name: str) -> str:
    """校验时间/空间分支模式是否合法。"""
    valid = {"sincos_linear", "sincos_mlp", "sincos_film"}
    if mode not in valid:
        raise ValueError(f"{name} 非法: {mode}, 期望 {sorted(valid)}")
    return mode


def check_expected_hw(hw: Tuple[int, int], expected_h: int, expected_w: int) -> None:
    """校验特征图网格尺寸是否符合计划假设。

    训练中很多设置都默认依赖固定 HxW：
    - token budget 估算
    - spatial encoding 的长度
    - batch/group 规模推导

    所以一旦真实尺寸和计划不一致，最好在这里尽早报错，而不是让后续 silently 偏掉。
    """
    h, w = int(hw[0]), int(hw[1])
    if h != int(expected_h) or w != int(expected_w):
        raise ValueError(f"特征图尺寸不匹配: got=({h},{w}) expected=({expected_h},{expected_w})")
