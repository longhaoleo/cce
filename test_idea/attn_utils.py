"""
注意力相关的通用工具函数：token 索引定位、热力图生成与保存。
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# -----------------------------
# 1) Token 索引定位（支持子词）
# -----------------------------
def find_subsequence_indices(full: List[int], sub: List[int]) -> List[int]:
    """在 full 中查找 sub（连续子序列），返回 sub 覆盖到的 full 索引列表（若多处匹配则取第一处）。"""
    if len(sub) == 0:
        return []
    for i in range(len(full) - len(sub) + 1):
        if full[i : i + len(sub)] == sub:
            return list(range(i, i + len(sub)))
    return []


def get_token_indices_for_phrase(
    tokenizer,
    prompt: str,
    phrase: str,
    max_length: int = 77,
) -> Tuple[List[int], Dict]:
    """
    返回 phrase 在 prompt token 序列中的索引列表（支持 phrase 被分成多个 token）。
    同时返回 debug 信息（tokens、ids 等）。
    """
    enc_prompt = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prompt_ids = enc_prompt.input_ids[0].tolist()

    enc_phrase = tokenizer(
        phrase,
        add_special_tokens=False,
        return_tensors="pt",
    )
    phrase_ids = enc_phrase.input_ids[0].tolist()

    indices = find_subsequence_indices(prompt_ids, phrase_ids)

    debug = {
        "prompt": prompt,
        "phrase": phrase,
        "prompt_ids": prompt_ids,
        "phrase_ids": phrase_ids,
        "prompt_tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
        "phrase_tokens": tokenizer.convert_ids_to_tokens(phrase_ids),
        "indices": indices,
    }
    return indices, debug


# -----------------------------
# 2) 热力图/掩膜可视化与保存
# -----------------------------
def to_colormap_uint8(x: np.ndarray) -> np.ndarray:
    """
    将二维数组归一化后映射到伪彩色（红黄白趋势）。
    返回 uint8 的 (H,W,3)。
    """
    x = x.astype(np.float32)
    x = x - x.min()
    denom = (x.max() + 1e-8)
    x = x / denom

    r = np.clip(2.0 * x, 0, 1)
    g = np.clip(2.0 * (x - 0.5), 0, 1)
    b = np.clip(1.0 - 2.0 * x, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return rgb


def save_heatmap_exact_resolution(attn_2d: np.ndarray, path: str):
    """
    保存严格像素分辨率等于 (H,W) 的热力图 PNG。
    """
    rgb = to_colormap_uint8(attn_2d)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def save_heatmap(path: str, arr_2d: np.ndarray):
    """
    保存热力图（伪彩色）。
    """
    img = Image.fromarray(to_colormap_uint8(arr_2d), mode="RGB")
    img.save(path)


def save_mask(path: str, mask_2d: np.ndarray):
    """
    保存二值 mask（0/1）为灰度图。
    """
    m = (mask_2d * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(m, mode="L")
    img.save(path)
