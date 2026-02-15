"""实验 54（子实验）：特定特征的因果干预（Injection / Ablation）。

说明
----
历史上这里叫 exp04，但现在仓库入口统一成 exp54（干预套件）。
这个文件保留“单窗口 baseline vs steered”的最小实现，供 exp05/06/07 复用。
"""

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
    feature_ids: List[int],
    feature_scales: List[float],
) -> Tuple[List[int], List[int], List[float]]:
    """从缓存中提取“所选特征集合”的加权综合激活曲线 c_t。

    约定：
    - 先对 tokens 求均值，得到每步的 [n_features] 平均激活向量
    - 再对指定 feature_ids 做加权求和，得到每步一个标量曲线
    """
    extractor = DeltaExtractor()
    deltas = extractor.extract(block=block, cache=cache, timesteps=timesteps)
    steps, ts, vals = [], [], []
    p = next(sae.parameters())
    ids = [int(x) for x in feature_ids]
    scales = [float(x) for x in feature_scales]
    for item in deltas:
        x = item.x.to(device=p.device, dtype=p.dtype)
        z = sae.encode(x)
        mu = z.mean(dim=0)  # [n_features]
        n_feat = int(mu.shape[0])
        acc = 0.0
        for fid, sc in zip(ids, scales):
            if 0 <= fid < n_feat:
                acc += float(mu[fid].item()) * float(sc)
        val = acc
        steps.append(int(item.step_idx))
        ts.append(int(item.timestep))
        vals.append(val)
    return steps, ts, vals


def run_exp54_causal_intervention(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    output_dir: str,
) -> None:
    """
    执行“单窗口因果干预”：对一组特征做注入/擦除，并输出对比图与曲线。

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

    # 统一解析“单特征/多特征”参数
    feature_ids = [int(x) for x in int_cfg.feature_ids]
    if not feature_ids:
        raise ValueError("feature_ids 不能为空。")
    if int_cfg.feature_scales:
        if len(int_cfg.feature_scales) != len(feature_ids):
            raise ValueError("int_feature_scales 长度必须与 int_feature_ids 相同。")
        feature_scales = [float(x) for x in int_cfg.feature_scales]
    else:
        feature_scales = [1.0 for _ in feature_ids]

    baseline_img = None
    baseline_curve = None
    timesteps_ref: List[int] = []
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
        feature_ids=feature_ids,
        feature_scales=feature_scales,
    )

    spec = InterventionSpec(
        block=block,
        feature_ids=tuple(feature_ids),
        feature_scales=tuple(feature_scales),
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
        feature_ids=feature_ids,
        feature_scales=feature_scales,
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
                "feature_ids",
                "feature_scales",
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
                        "feature_ids": " ".join(str(x) for x in feature_ids),
                        "feature_scales": " ".join(str(x) for x in feature_scales),
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
                        "feature_ids": " ".join(str(x) for x in feature_ids),
                        "feature_scales": " ".join(str(x) for x in feature_scales),
                        "baseline_value": vals_b[i],
                        "steered_value": vals_s[i],
                        "delta": vals_s[i] - vals_b[i],
                    }
                )
    print(f"已保存曲线 CSV: {csv_path}")

    fid_tag = f"f{feature_ids[0]}_k{len(feature_ids)}" if len(feature_ids) > 1 else f"f{feature_ids[0]}"
    curve_path = os.path.join(output_dir, f"intervention_curve_{safe_name(block)}_{fid_tag}.png")
    plt.figure(figsize=(10, 4))
    if baseline_curve is not None:
        _, _, vals_b = baseline_curve
        plt.plot(range(len(vals_b)), vals_b, label="baseline", marker="o")
    plt.plot(range(len(vals_s)), vals_s, label="steered", marker="s")
    plt.xlabel("step_idx")
    plt.ylabel("feature activation")
    plt.title(
        f"Intervention Curve | block={block}\n"
        f"mode={int_cfg.mode} features={feature_ids} scales={feature_scales} global_scale={float(int_cfg.scale)}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()
    print(f"已保存曲线图: {curve_path}")
