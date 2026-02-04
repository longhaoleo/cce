"""
mask 相关的通用工具函数：尺寸调整与合并。
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int], mode: str = "nearest") -> np.ndarray:
    """
    将 2D mask resize 到目标分辨率。
    """
    if mask.shape == target_hw:
        return mask
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=target_hw, mode=mode)
    return t.squeeze(0).squeeze(0).cpu().numpy()


def merge_masks(base: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """
    将两个 mask 叠加并裁剪到 [0,1]。
    """
    return np.clip(base + extra, 0, 1)
