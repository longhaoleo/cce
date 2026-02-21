"""通用工具函数。"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """将张量归一化到 [0, 1]。"""
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


def safe_name(s: str) -> str:
    """把任意字符串转成适合文件名的形式。"""
    s = s.strip()
    s = re.sub(r"[^\w\.\-]+", "_", s, flags=re.UNICODE)
    return s[:180] if len(s) > 180 else s


def block_short_name(block: str) -> str:
    """将 block 名称压缩成短名（用于 out_concept_dict_<block>）。"""
    name = str(block).strip()
    m = re.search(r"up_blocks\.(\d+)\.attentions\.(\d+)", name)
    if m:
        return f"up.{m.group(1)}.{m.group(2)}"
    m = re.search(r"down_blocks\.(\d+)\.attentions\.(\d+)", name)
    if m:
        return f"down.{m.group(1)}.{m.group(2)}"
    m = re.search(r"mid_block\.attentions\.(\d+)", name)
    if m:
        return f"mid.{m.group(1)}"
    return safe_name(name)


def extract_first_image(output: Any) -> Optional[Image.Image]:
    """从 pipeline 输出里安全提取第一张 PIL 图像。"""
    if output is None:
        return None
    images = getattr(output, "images", None)
    if images is not None and len(images) > 0:
        img = images[0]
        if isinstance(img, Image.Image):
            return img
    if isinstance(output, list) and output and isinstance(output[0], Image.Image):
        return output[0]
    return None


def overlay_heatmap(
    heat_2d: torch.Tensor,
    *,
    out_path: str,
    title: str,
    base_image: Optional[Image.Image] = None,
    alpha: float = 0.75,
) -> None:
    """
    保存单步热力图。

    - 有底图时：将热力图上采样后叠加；
    - 无底图时：直接绘制热力图。
    """
    plt.figure(figsize=(4, 4))
    if base_image is not None:
        bw, bh = base_image.size
        base_np = np.asarray(base_image.convert("RGB"))
        up = F.interpolate(
            heat_2d[None, None].detach().float().cpu(),
            size=(bh, bw),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        plt.imshow(base_np)
        plt.imshow(up, cmap="turbo", alpha=float(alpha), vmin=0.0, vmax=1.0)
    else:
        plt.imshow(heat_2d.detach().float().cpu().numpy(), cmap="turbo", vmin=0.0, vmax=1.0)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
