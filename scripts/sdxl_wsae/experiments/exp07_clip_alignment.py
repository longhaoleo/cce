"""实验 7：CLIP 对齐分数（Alignment Score）评估。"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig
from ..utils import ensure_dir
from .exp04_causal_intervention import run_exp04_causal_intervention


@dataclass
class ClipEvalConfig:
    """CLIP 评估配置。"""

    target_text: str = "red"
    preserve_text: str = "car"
    model_name: str = "openai/clip-vit-large-patch14"


def _compute_clip_score(
    *,
    image: Image.Image,
    text: str,
    model_name: str,
    device: str,
) -> Optional[float]:
    """计算单图单文本的 CLIP 相似度分数。"""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return None

    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
    except Exception:
        return None

    with torch.no_grad():
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        score = float(outputs.logits_per_image[0, 0].item())
    return score


def _evaluate_images(
    *,
    baseline_path: str,
    steered_path: str,
    clip_cfg: ClipEvalConfig,
    output_csv: str,
) -> None:
    """对 baseline/steered 图片计算 CLIP 分数并输出 CSV。"""
    rows: List[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for tag, img_path in [("baseline", baseline_path), ("steered", steered_path)]:
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        score_target = _compute_clip_score(
            image=img,
            text=clip_cfg.target_text,
            model_name=clip_cfg.model_name,
            device=device,
        )
        score_preserve = _compute_clip_score(
            image=img,
            text=clip_cfg.preserve_text,
            model_name=clip_cfg.model_name,
            device=device,
        )
        rows.append(
            {
                "image_tag": tag,
                "image_path": img_path,
                "target_text": clip_cfg.target_text,
                "target_score": score_target if score_target is not None else "",
                "preserve_text": clip_cfg.preserve_text,
                "preserve_score": score_preserve if score_preserve is not None else "",
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image_tag",
            "image_path",
            "target_text",
            "target_score",
            "preserve_text",
            "preserve_score",
        ])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def run_exp07_clip_alignment(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    clip_cfg: ClipEvalConfig,
    output_dir: str,
) -> None:
    """
    实验 7：
    1) 先跑一次因果干预得到 baseline/steered；
    2) 计算 CLIP(target) 与 CLIP(preserve) 分数。
    """
    ensure_dir(output_dir)
    subdir = os.path.join(output_dir, "exp07_clip_alignment")
    ensure_dir(subdir)

    run_exp04_causal_intervention(model_cfg, sae_cfg, run_cfg, int_cfg, subdir)

    baseline_path = os.path.join(subdir, "intervention_baseline.png")
    steered_path = os.path.join(subdir, "intervention_steered.png")
    clip_csv = os.path.join(subdir, "clip_alignment_scores.csv")
    _evaluate_images(
        baseline_path=baseline_path,
        steered_path=steered_path,
        clip_cfg=clip_cfg,
        output_csv=clip_csv,
    )
    print(f"实验 7 完成，CLIP 分数已输出: {clip_csv}")
