"""Shared SAE 产物目录布局。"""

from __future__ import annotations

from pathlib import Path


LAYOUT_DIRS = {
    "concept_dig": "concept-dig",
    "concept_dig_freq": "concept-dig-freq",
    "blacklist": "blacklist",
    "feature_freq": "feature-freq",
}


def layout_subdir(*, sae_root: str, kind: str) -> str:
    """在 SAE 根目录下解析某类产物目录。"""
    root = str(sae_root).strip()
    if not root:
        raise ValueError("sae_root 不能为空。")
    if kind not in LAYOUT_DIRS:
        raise KeyError(f"未知 layout kind: {kind}")
    return str((Path(root).expanduser() / LAYOUT_DIRS[kind]).resolve())


def maybe_use_sae_layout(*, path_value: str, sae_root: str, legacy_default: str, kind: str) -> str:
    """当用户只传 SAE 根目录时，把旧默认根目录映射到统一布局。"""
    raw = str(path_value).strip()
    root = str(sae_root).strip()
    if not root:
        return raw
    if raw != str(legacy_default).strip():
        return raw
    return layout_subdir(sae_root=root, kind=kind)
