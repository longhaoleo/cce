"""
实验 54：统一干预入口（时间窗差异对比 / “蝴蝶效应”）。


这些实验的核心都是：
1) 选一个 block + 一组 SAE feature_ids（单特征就是长度=1）
2) 构造干预 hook（Injection/Ablation）
3) 在扩散轨迹的某个时间窗口内对该 block 输出做干预
4) 输出生成图 + 该特征随时间的曲线

因此这里把它们合并成一个入口 exp54：
- main 干预：使用 int_cfg.mode + int_cfg.t_start/t_end（等价于原 exp04）
- early/late 干预：仅当 main 的 mode 是 injection 时运行，窗口来自 tw_cfg

本文件同时提供一个“单窗口 baseline vs steered”的子实验函数：
- `run_exp54_causal_intervention(...)`
它会被 exp05/exp06/exp07/exp21 复用（避免重复代码）。

输入参数来源（均已在 CLI 里存在）：
- int_cfg: 通过 --int_block/--int_feature_ids/--int_mode/--int_scale/--int_t_start/--int_t_end/... 指定
- tw_cfg: 通过 --early_start/--early_end/--late_start/--late_end 指定

输出目录
--------
输出到 `output_dir/exp54_intervention_suite/`：
- baseline.png（若启用 baseline）
- main_injection.png 或 main_ablation.png
- early_injection.png / late_injection.png（仅当 main=inj）
- compare_*.png（把可用的图片横向拼接，便于快速肉眼对比）
- suite_curve_{block}_f{feature}.csv
- suite_curve_{block}_f{feature}.png

说明
----
这里“特征曲线”是这样算的：
- 取 delta = h_out - h_in（同一个 hookpoint 的输出减输入）
- 用 SAE.encode(delta_tokens) 得到 token 级激活
- 对 tokens 取均值，得到该 step 的全局激活值 c_t

注意
----
这里的“干预”本身使用的是 `x` 的 SAE 编码系数 `c_i(x)` 来做对称注入/擦除：
- injection: x <- x + scale * (c_i(x) * d_i)
- ablation:  x <- x - scale * (c_i(x) * d_i)
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image

from ..configs import CausalInterventionConfig, RunConfig, SAEConfig, TemporalWindowConfig
from ..core.intervention import InterventionSpec, build_feature_intervention_hook
from ..core.session import SDXLExperimentSession
from ..utils import ensure_dir, extract_first_image, safe_name
from .shared_prepare import DeltaExtractor


def _concat_images_h(images: List[Image.Image]) -> Image.Image:
    """将多张图片水平拼接。"""
    assert images
    w = sum(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
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
    """从缓存中提取“所选特征集合”的加权综合激活曲线 c_t。"""
    extractor = DeltaExtractor()
    deltas = extractor.extract(block=block, cache=cache, timesteps=timesteps)

    steps: List[int] = []
    ts: List[int] = []
    vals: List[float] = []

    p = next(sae.parameters())
    ids = [int(x) for x in feature_ids]
    scales = [float(x) for x in feature_scales]
    for item in deltas:
        x = item.x.to(device=p.device, dtype=p.dtype)  # [tokens, d_model]
        z = sae.encode(x)  # [tokens, n_features]
        mu = z.mean(dim=0)
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
    """exp54 的“单窗口因果干预”子实验（供 exp05/06/07/21 复用）。

    这就是历史上的 exp04：baseline vs steered。

    输出文件（都写到 output_dir 下）：
    - intervention_baseline.png（可选，由 int_cfg.compare_baseline 控制）
    - intervention_steered.png
    - intervention_compare.png（可选：有 baseline 才会输出）
    - intervention_feature_curve.csv
    - intervention_curve_{block}_f{...}.png
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
    if bool(getattr(int_cfg, "compare_baseline", True)):
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

        timesteps = session.scheduler_timesteps(session.pipe)
        baseline_curve = _feature_curve_from_cache(
            cache=cache_base,
            timesteps=timesteps,
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
        spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
        mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
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
        compare = _concat_images_h([baseline_img, steered_img])
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
        f"mode={int_cfg.mode} features={feature_ids} scales={feature_scales} global_scale={float(int_cfg.scale)} "
        f"mask={str(getattr(int_cfg, 'spatial_mask', 'none'))} sigma={float(getattr(int_cfg, 'mask_sigma', 0.25))}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()
    print(f"已保存曲线图: {curve_path}")


def run_exp54_intervention_suite(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    tw_cfg: TemporalWindowConfig,
    output_dir: str,
) -> None:
    """执行 exp54：baseline + main +（可选）early/late 对比。"""
    ensure_dir(output_dir)
    root = os.path.join(output_dir, "exp54_intervention_suite")
    ensure_dir(root)

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

    # 1) Baseline：不加 hook，正常采样并缓存该 block 的输入/输出（可选）
    baseline_img = None
    baseline_curve = None
    if bool(getattr(int_cfg, "compare_baseline", True)):
        out_base, cache_base = session.run_with_cache(
            run_cfg,
            positions_to_cache=[block],
            save_input=True,
            save_output=True,
            output_type="pil",
        )
        baseline_img = extract_first_image(out_base)
        if baseline_img is not None:
            pth = os.path.join(root, "baseline.png")
            baseline_img.save(pth)
            print(f"已保存 baseline: {pth}")

        timesteps = session.scheduler_timesteps(session.pipe)
        baseline_curve = _feature_curve_from_cache(
            cache=cache_base,
            timesteps=timesteps,
            block=block,
            sae=sae,
            feature_ids=feature_ids,
            feature_scales=feature_scales,
        )

    # 2) 通用：跑一次“带 hook 的推理”，并返回生成图 + 曲线
    def _run_one(*, name: str, spec: InterventionSpec):
        hook = build_feature_intervention_hook(pipe=session.pipe, sae=sae, spec=spec)
        out, cache = session.run_with_hooks_and_cache(
            run_cfg,
            position_hook_dict={block: hook},
            positions_to_cache=[block],
            save_input=True,
            save_output=True,
            output_type="pil",
        )
        img = extract_first_image(out)
        if img is not None:
            pth = os.path.join(root, f"{name}.png")
            img.save(pth)
            print(f"已保存 {name}: {pth}")
        timesteps = session.scheduler_timesteps(session.pipe)
        curve = _feature_curve_from_cache(
            cache=cache,
            timesteps=timesteps,
            block=block,
            sae=sae,
            feature_ids=feature_ids,
            feature_scales=feature_scales,
        )
        return img, curve

    # 2.1) main：等价于原 exp04（mode + t_start/t_end）
    main_mode = str(int_cfg.mode).lower()
    main_name = f"main_{main_mode}"
    main_spec = InterventionSpec(
        block=block,
        feature_ids=tuple(feature_ids),
        feature_scales=tuple(feature_scales),
        mode=main_mode,  # injection | ablation
        scale=float(int_cfg.scale),
        spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
        mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
        t_start=int(int_cfg.t_start),
        t_end=int(int_cfg.t_end),
        step_start=int_cfg.step_start,
        step_end=int_cfg.step_end,
        apply_only_conditional=True,
    )
    main_img, main_curve = _run_one(name=main_name, spec=main_spec)

    # 2.2) early/late：只有当 main 是 injection 时才跑
    early_img = late_img = None
    early_curve = late_curve = None
    if main_mode == "injection":
        early_spec = InterventionSpec(
            block=block,
            feature_ids=tuple(feature_ids),
            feature_scales=tuple(feature_scales),
            mode="injection",
            scale=float(int_cfg.scale),
            spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
            mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
            t_start=int(tw_cfg.early_start),
            t_end=int(tw_cfg.early_end),
            step_start=int_cfg.step_start,
            step_end=int_cfg.step_end,
            apply_only_conditional=True,
        )
        late_spec = InterventionSpec(
            block=block,
            feature_ids=tuple(feature_ids),
            feature_scales=tuple(feature_scales),
            mode="injection",
            scale=float(int_cfg.scale),
            spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
            mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
            t_start=int(tw_cfg.late_start),
            t_end=int(tw_cfg.late_end),
            step_start=int_cfg.step_start,
            step_end=int_cfg.step_end,
            apply_only_conditional=True,
        )
        early_img, early_curve = _run_one(name="early_injection", spec=early_spec)
        late_img, late_curve = _run_one(name="late_injection", spec=late_spec)

    # 3) 拼接对比图：把可用的图片横向拼接
    compare_imgs: List[Tuple[str, Image.Image]] = []
    if baseline_img is not None:
        compare_imgs.append(("baseline", baseline_img))
    if main_img is not None:
        compare_imgs.append((main_name, main_img))
    if early_img is not None:
        compare_imgs.append(("early", early_img))
    if late_img is not None:
        compare_imgs.append(("late", late_img))
    if len(compare_imgs) >= 2:
        concat = _concat_images_h([im for _, im in compare_imgs])
        tags = "_".join(tag for tag, _ in compare_imgs)
        out_path = os.path.join(root, f"compare_{tags}.png")
        concat.save(out_path)
        print(f"已保存拼接对比图: {out_path}")

    # 4) 曲线导出：统一写一个 CSV，并画一张对比曲线
    curves: Dict[str, Tuple[List[int], List[int], List[float]]] = {main_name: main_curve}
    if baseline_curve is not None:
        curves["baseline"] = baseline_curve
    if early_curve is not None:
        curves["early_injection"] = early_curve
    if late_curve is not None:
        curves["late_injection"] = late_curve

    n = min(len(v[2]) for v in curves.values())
    ref_key = next(iter(curves.keys()))
    steps_ref, ts_ref, _ = curves[ref_key]

    fid_tag = f"f{feature_ids[0]}_k{len(feature_ids)}" if len(feature_ids) > 1 else f"f{feature_ids[0]}"
    csv_path = os.path.join(root, f"suite_curve_{safe_name(block)}_{fid_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["step_idx", "timestep", "feature_ids", "feature_scales"] + [f"{k}_value" for k in curves.keys()]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            row = {
                "step_idx": int(steps_ref[i]),
                "timestep": int(ts_ref[i]),
                "feature_ids": " ".join(str(x) for x in feature_ids),
                "feature_scales": " ".join(str(x) for x in feature_scales),
            }
            for k, (_, _, vals) in curves.items():
                row[f"{k}_value"] = float(vals[i])
            w.writerow(row)
    print(f"已保存曲线 CSV: {csv_path}")

    fig_path = os.path.join(root, f"suite_curve_{safe_name(block)}_{fid_tag}.png")
    plt.figure(figsize=(10, 4))
    markers = ["o", "s", "^", "x", "d"]
    for idx, (k, (_, _, vals)) in enumerate(curves.items()):
        plt.plot(range(n), vals[:n], label=k, marker=markers[idx % len(markers)])
    plt.xlabel("step_idx")
    plt.ylabel("feature activation")
    plt.title(
        f"exp54 Intervention Suite | block={block} features={feature_ids}\n"
        f"main={main_mode} scale={float(int_cfg.scale)} "
        f"main=[{int(int_cfg.t_start)},{int(int_cfg.t_end)}] "
        f"early=[{int(tw_cfg.early_start)},{int(tw_cfg.early_end)}] "
        f"late=[{int(tw_cfg.late_start)},{int(tw_cfg.late_end)}]"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"已保存曲线图: {fig_path}")

    print(f"实验 54 完成，输出目录: {root}")
