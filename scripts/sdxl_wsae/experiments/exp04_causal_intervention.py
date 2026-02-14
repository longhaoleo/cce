"""实验 4：特定特征的因果干预（Injection / Ablation）。"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig
from ..core.intervention import InterventionSpec, build_feature_intervention_hook
from ..core.session import SDXLExperimentSession
from .shared_prepare import DeltaExtractor
from ..utils import ensure_dir, extract_first_image, safe_name


def _concat_images_h(left: Image.Image, right: Image.Image) -> Image.Image:
    """将两张图片左右拼接。"""
    w = left.width + right.width
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


@torch.no_grad()
def _feature_curve_from_cache(
    *,
    cache,
    timesteps: List[int],
    block: str,
    sae: torch.nn.Module,
    feature_id: int,
) -> Tuple[List[int], List[int], List[float]]:
    """从缓存中提取某一特征的时序激活曲线 c_t。"""
    extractor = DeltaExtractor()
    deltas = extractor.extract(block=block, cache=cache, timesteps=timesteps)
    steps, ts, vals = [], [], []
    p = next(sae.parameters())
    fid = int(feature_id)
    for item in deltas:
        x = item.x.to(device=p.device, dtype=p.dtype)
        z = sae.encode(x)
        if fid < 0 or fid >= int(z.shape[1]):
            val = 0.0
        else:
            val = float(z[:, fid].mean().item())
        steps.append(int(item.step_idx))
        ts.append(int(item.timestep))
        vals.append(val)
    return steps, ts, vals


def run_exp04_causal_intervention(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    output_dir: str,
) -> None:
    """
    执行实验 4：对单个特征做注入/擦除，并输出对比图与曲线。

    结果文件：
    - `intervention_baseline.png` / `intervention_steered.png`
    - `intervention_compare.png`
    - `intervention_feature_curve.csv`
    - `intervention_curve_*.png`
    """
    ensure_dir(output_dir)
    block = str(int_cfg.block)
    session = SDXLExperimentSession(model_cfg, sae_cfg)
    session.load_saes([block])
    sae = session.get_sae(block)

    baseline_img = None
    baseline_curve = None
    timesteps_ref: List[int] = []
    if bool(int_cfg.compare_baseline):
        out_base, cache_base = session.run_with_cache(
            run_cfg,
            positions_to_cache=[block],
            save_input=True,
            save_output=True,
            output_type="pil",
        )
        baseline_img = extract_first_image(out_base)
        if baseline_img is not None:
            baseline_path = os.path.join(output_dir, "intervention_baseline.png")
            baseline_img.save(baseline_path)
            print(f"已保存 baseline: {baseline_path}")
        timesteps_ref = session.scheduler_timesteps(session.pipe)
        baseline_curve = _feature_curve_from_cache(
            cache=cache_base,
            timesteps=timesteps_ref,
            block=block,
            sae=sae,
            feature_id=int_cfg.feature_id,
        )

    spec = InterventionSpec(
        block=block,
        feature_id=int(int_cfg.feature_id),
        mode=str(int_cfg.mode),
        scale=float(int_cfg.scale),
        t_start=int(int_cfg.t_start),
        t_end=int(int_cfg.t_end),
        step_start=int_cfg.step_start,
        step_end=int_cfg.step_end,
        apply_only_conditional=True,
    )
    hook = build_feature_intervention_hook(pipe=session.pipe, sae=sae, spec=spec)

    out_steer, cache_steer = session.run_with_hooks_and_cache(
        run_cfg,
        position_hook_dict={block: hook},
        positions_to_cache=[block],
        save_input=True,
        save_output=True,
        output_type="pil",
    )
    steered_img = extract_first_image(out_steer)
    if steered_img is not None:
        steered_path = os.path.join(output_dir, "intervention_steered.png")
        steered_img.save(steered_path)
        print(f"已保存 steered: {steered_path}")

    timesteps = session.scheduler_timesteps(session.pipe)
    steps_s, ts_s, vals_s = _feature_curve_from_cache(
        cache=cache_steer,
        timesteps=timesteps,
        block=block,
        sae=sae,
        feature_id=int_cfg.feature_id,
    )

    if baseline_img is not None and steered_img is not None:
        compare = _concat_images_h(baseline_img, steered_img)
        compare_path = os.path.join(output_dir, "intervention_compare.png")
        compare.save(compare_path)
        print(f"已保存对比图: {compare_path}")

    csv_path = os.path.join(output_dir, "intervention_feature_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "timestep",
                "feature_id",
                "baseline_value",
                "steered_value",
                "delta",
            ],
        )
        writer.writeheader()
        if baseline_curve is None:
            for i in range(len(steps_s)):
                writer.writerow(
                    {
                        "step_idx": steps_s[i],
                        "timestep": ts_s[i],
                        "feature_id": int(int_cfg.feature_id),
                        "baseline_value": "",
                        "steered_value": vals_s[i],
                        "delta": "",
                    }
                )
        else:
            steps_b, ts_b, vals_b = baseline_curve
            n = min(len(vals_b), len(vals_s))
            for i in range(n):
                writer.writerow(
                    {
                        "step_idx": steps_s[i],
                        "timestep": ts_s[i],
                        "feature_id": int(int_cfg.feature_id),
                        "baseline_value": vals_b[i],
                        "steered_value": vals_s[i],
                        "delta": vals_s[i] - vals_b[i],
                    }
                )
    print(f"已保存曲线 CSV: {csv_path}")

    curve_path = os.path.join(
        output_dir,
        f"intervention_curve_{safe_name(block)}_f{int(int_cfg.feature_id)}.png",
    )
    plt.figure(figsize=(10, 4))
    if baseline_curve is not None:
        _, _, vals_b = baseline_curve
        plt.plot(range(len(vals_b)), vals_b, label="baseline", marker="o")
    plt.plot(range(len(vals_s)), vals_s, label="steered", marker="s")
    plt.xlabel("step_idx")
    plt.ylabel("feature activation")
    plt.title(
        f"Intervention Curve | block={block}\n"
        f"mode={int_cfg.mode} feature={int(int_cfg.feature_id)} scale={float(int_cfg.scale)}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()
    print(f"已保存曲线图: {curve_path}")
