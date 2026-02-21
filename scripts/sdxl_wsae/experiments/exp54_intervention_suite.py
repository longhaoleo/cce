"""
实验 54：统一干预入口（时间窗差异对比 / “蝴蝶效应”）。


这些实验的核心都是：
1) 选一个 block + 从 rank_csv 里取 top-k 特征
2) 构造干预 hook（Injection/Ablation）
3) 在扩散轨迹的某个时间窗口内对该 block 输出做干预
4) 输出生成图 + 该特征随时间的曲线

因此这里把它们合并成一个入口 exp54：
- main 干预：使用 int_cfg.mode + int_cfg.t_start/t_end（等价于原 exp04）
- early/late 干预：仅当 main 的 mode 是 injection 时运行，窗口来自 tw_cfg

本文件同时提供一个“单窗口 baseline vs steered”的子实验函数：
- `run_exp54_causal_intervention(...)`
它会被 exp05/exp06/exp07 复用（避免重复代码）。

输入参数来源（均已在 CLI 里存在）：
- int_cfg: 通过 --int_block/--int_feature_top_k/--int_feature_rank_csv/--int_mode/--int_scale/--int_t_start/--int_t_end/... 指定
- tw_cfg: 通过 --early_start/--early_end/--late_start/--late_end 指定

输出目录
--------
输出到 `output_dir/exp54_intervention_suite/`：
- baseline.png（若启用 baseline）
- main_injection.png 或 main_ablation.png
- early_{mode}.png / late_{mode}.png（mode=injection/ablation）
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

默认从 exp53 读取系数曲线（并自动选 top-k 特征）
----------------------------
如果你希望更细粒度地控制“在不同 t/step 注入多少”，需要：
- `--int_coeff_csv out_concept_dict_<block_short>/<concept>/feature_time_scores.csv`
特征来源改为：
- `--int_feature_top_k 10`
默认从 coeff_csv 同目录自动读取 `top_positive_features.csv`。

此时干预系数不再来自当前 x 的 encode，而是来自 exp53 统计的“按 step 的平均激活曲线”。
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
from ..utils import ensure_dir, extract_first_image, safe_name, block_short_name
from .shared_prepare import DeltaExtractor


def _load_topk_feature_ids(csv_path: str, k: int) -> List[int]:
    """从 top_positive_features.csv 读取前 K 个 feature_id。"""
    path = os.path.expanduser(str(csv_path))
    if not path:
        raise ValueError("feature_rank_csv 不能为空。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_rank_csv 不存在: {path}")
    k = int(k)
    if k <= 0:
        raise ValueError("feature_top_k 必须 > 0。")

    rows: List[Tuple[int, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if not {"feature_id", "score"}.issubset(fields):
            raise ValueError(f"rank csv 缺少列: feature_id/score, got={r.fieldnames}")
        for row in r:
            try:
                fid = int(row["feature_id"])
                score = float(row["score"])
            except Exception:
                continue
            rows.append((fid, score))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [fid for fid, _ in rows[:k]]


def _resolve_feature_ids_and_scales(
    *,
    block: str,
    targetconcept: str,
    feature_top_k: int,
) -> Tuple[List[int], List[float]]:
    """解析特征 id 与 scale（从 block 专属目录的 rank_csv 取 top-k，scale=1）。"""
    block_tag = block_short_name(block)
    base_dir = os.path.join(f"out_concept_dict_{block_tag}", targetconcept)
    rank_csv = os.path.join(base_dir, "top_positive_features.csv")
    feature_ids = _load_topk_feature_ids(rank_csv, int(feature_top_k))

    if not feature_ids:
        raise ValueError("feature_ids 为空，无法从 rank_csv 获取 top-k。")
    feature_scales = [1.0 for _ in feature_ids]
    return feature_ids, feature_scales


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


def _load_coeff_by_step_from_exp53_csv(
    *,
    csv_path: str,
    feature_ids: List[int],
) -> Dict[int, torch.Tensor]:
    """从 exp53 的 feature_time_scores.csv 读取“按 step 的系数表”。

    仅支持长表格式：step_idx, feature_id, pos_mu/neg_mu/diff
    """
    path = os.path.expanduser(str(csv_path))
    if not path:
        raise ValueError("int_coeff_csv 不能为空。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"int_coeff_csv 不存在: {path}")

    col = "diff"

    feats = [int(x) for x in feature_ids]
    fid_to_pos = {fid: i for i, fid in enumerate(feats)}

    tmp: Dict[int, Dict[int, float]] = {}  # step -> fid -> val
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])

        if not {"step_idx", "feature_id", col}.issubset(fields):
            raise ValueError(f"coeff csv 缺少列: step_idx/feature_id/{col}, got={r.fieldnames}")
        for row in r:
            try:
                step = int(row["step_idx"])
                fid = int(row["feature_id"])
                if fid not in fid_to_pos:
                    continue
                val = float(row[col])
            except Exception:
                continue
            tmp.setdefault(step, {})[fid] = val

    out: Dict[int, torch.Tensor] = {}
    for step, fid_map in tmp.items():
        vec = torch.zeros(len(feats), dtype=torch.float32)
        for fid, v in fid_map.items():
            vec[fid_to_pos[int(fid)]] = float(v)
        out[int(step)] = vec
    return out


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
    """exp54 单窗口干预，支持多 block 同时 hook。"""
    ensure_dir(output_dir)
    blocks = list(getattr(int_cfg, "blocks", ()) or [])
    if not blocks:
        raise ValueError("int_cfg.blocks 不能为空。")

    session = SDXLExperimentSession(model_cfg, sae_cfg)
    session.load_saes(blocks)

    feats_by_block: Dict[str, Tuple[List[int], List[float]]] = {}
    coeffs_by_block: Dict[str, Dict[int, torch.Tensor]] = {}
    for block in blocks:
        f_ids, f_scales = _resolve_feature_ids_and_scales(
            block=block,
            targetconcept=str(getattr(int_cfg, "targetconcept", "")),
            feature_top_k=int(getattr(int_cfg, "feature_top_k", 0)),
        )
        base_dir = os.path.join(f"out_concept_dict_{block_short_name(block)}", str(getattr(int_cfg, "targetconcept", "")))
        coeffs = _load_coeff_by_step_from_exp53_csv(
            csv_path=os.path.join(base_dir, "feature_time_scores.csv"),
            feature_ids=f_ids,
        )
        feats_by_block[block] = (f_ids, f_scales)
        coeffs_by_block[block] = coeffs

    baseline_img = None
    baseline_curves: Dict[str, Tuple[List[int], List[int], List[float]]] = {}
    if bool(getattr(int_cfg, "compare_baseline", True)):
        out_base, cache_base = session.run_with_cache(
            run_cfg,
            positions_to_cache=blocks,
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
        for block in blocks:
            sae = session.get_sae(block)
            f_ids, f_scales = feats_by_block[block]
            baseline_curves[block] = _feature_curve_from_cache(
                cache=cache_base,
                timesteps=timesteps,
                block=block,
                sae=sae,
                feature_ids=f_ids,
                feature_scales=f_scales,
            )

    hooks = {}
    for block in blocks:
        sae = session.get_sae(block)
        f_ids, f_scales = feats_by_block[block]
        spec = InterventionSpec(
            block=block,
            feature_ids=tuple(f_ids),
            feature_scales=tuple(f_scales),
            mode=str(int_cfg.mode),
            scale=float(int_cfg.scale),
            spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
            mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
            coeff_by_step=coeffs_by_block[block],
            t_start=int(int_cfg.t_start),
            t_end=int(int_cfg.t_end),
            step_start=int_cfg.step_start,
            step_end=int_cfg.step_end,
            apply_only_conditional=True,
        )
        hooks[block] = build_feature_intervention_hook(pipe=session.pipe, sae=sae, spec=spec)

    out_steer, cache_steer = session.run_with_hooks_and_cache(
        run_cfg,
        position_hook_dict=hooks,
        positions_to_cache=blocks,
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
    curves_steer: Dict[str, Tuple[List[int], List[int], List[float]]] = {}
    for block in blocks:
        sae = session.get_sae(block)
        f_ids, f_scales = feats_by_block[block]
        curves_steer[block] = _feature_curve_from_cache(
            cache=cache_steer,
            timesteps=timesteps,
            block=block,
            sae=sae,
            feature_ids=f_ids,
            feature_scales=f_scales,
        )

    if baseline_img is not None and steered_img is not None:
        compare = _concat_images_h([baseline_img, steered_img])
        compare_path = os.path.join(output_dir, "intervention_compare.png")
        compare.save(compare_path)
        print(f"已保存对比图: {compare_path}")

    for block in blocks:
        steps_s, ts_s, vals_s = curves_steer[block]
        steps_b, ts_b, vals_b = baseline_curves.get(block, ([], [], []))
        curve_csv = os.path.join(output_dir, f"intervention_curve_{safe_name(block)}.csv")
        with open(curve_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step_idx", "timestep", "steer_value", "baseline_value"])
            for (s, t, v_s), (_, _, v_b) in zip(zip(steps_s, ts_s, vals_s), zip(steps_b, ts_b, vals_b)):
                writer.writerow([s, t, v_s, v_b if baseline_curves else ""])
        print(f"已保存曲线: {curve_csv}")
def run_exp54_intervention_suite(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    int_cfg: CausalInterventionConfig,
    tw_cfg: TemporalWindowConfig,
    output_dir: str,
) -> None:
    """执行 exp54：baseline + main + early/late，对多个 block 同时干预。"""
    ensure_dir(output_dir)
    root = os.path.join(output_dir, "exp54_intervention_suite")
    ensure_dir(root)

    blocks = list(getattr(int_cfg, "blocks", ()) or [])
    if not blocks:
        raise ValueError("int_cfg.blocks 不能为空。")

    session = SDXLExperimentSession(model_cfg, sae_cfg)
    session.load_saes(blocks)

    feats_by_block: Dict[str, Tuple[List[int], List[float]]] = {}
    coeffs_by_block: Dict[str, Dict[int, torch.Tensor]] = {}
    for block in blocks:
        f_ids, f_scales = _resolve_feature_ids_and_scales(
            block=block,
            targetconcept=str(getattr(int_cfg, "targetconcept", "")),
            feature_top_k=int(getattr(int_cfg, "feature_top_k", 0)),
        )
        base_dir = os.path.join(f"out_concept_dict_{block_short_name(block)}", str(getattr(int_cfg, "targetconcept", "")))
        coeffs = _load_coeff_by_step_from_exp53_csv(
            csv_path=os.path.join(base_dir, "feature_time_scores.csv"),
            feature_ids=f_ids,
        )
        feats_by_block[block] = (f_ids, f_scales)
        coeffs_by_block[block] = coeffs

    # baseline（可选）
    baseline_img = None
    baseline_curves: Dict[str, Tuple[List[int], List[int], List[float]]] = {}
    if bool(getattr(int_cfg, "compare_baseline", True)):
        out_base, cache_base = session.run_with_cache(
            run_cfg,
            positions_to_cache=blocks,
            save_input=True,
            save_output=True,
            output_type="pil",
        )
        baseline_img = extract_first_image(out_base)
        if baseline_img is not None:
            baseline_img.save(os.path.join(root, "baseline.png"))
            print(f"已保存 baseline: {os.path.join(root, 'baseline.png')}")
        timesteps = session.scheduler_timesteps(session.pipe)
        for block in blocks:
            sae = session.get_sae(block)
            f_ids, f_scales = feats_by_block[block]
            baseline_curves[block] = _feature_curve_from_cache(
                cache=cache_base,
                timesteps=timesteps,
                block=block,
                sae=sae,
                feature_ids=f_ids,
                feature_scales=f_scales,
            )

    def _run_one(name: str, t_start: int, t_end: int):
        hooks = {}
        for block in blocks:
            sae = session.get_sae(block)
            f_ids, f_scales = feats_by_block[block]
            spec = InterventionSpec(
                block=block,
                feature_ids=tuple(f_ids),
                feature_scales=tuple(f_scales),
                mode=str(int_cfg.mode),
                scale=float(int_cfg.scale),
                spatial_mask=str(getattr(int_cfg, "spatial_mask", "none")),
                mask_sigma=float(getattr(int_cfg, "mask_sigma", 0.25)),
                coeff_by_step=coeffs_by_block[block],
                t_start=int(t_start),
                t_end=int(t_end),
                step_start=int_cfg.step_start,
                step_end=int_cfg.step_end,
                apply_only_conditional=True,
            )
            hooks[block] = build_feature_intervention_hook(pipe=session.pipe, sae=sae, spec=spec)

        out, cache = session.run_with_hooks_and_cache(
            run_cfg,
            position_hook_dict=hooks,
            positions_to_cache=blocks,
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
        curves: Dict[str, Tuple[List[int], List[int], List[float]]] = {}
        for block in blocks:
            sae = session.get_sae(block)
            f_ids, f_scales = feats_by_block[block]
            curves[block] = _feature_curve_from_cache(
                cache=cache,
                timesteps=timesteps,
                block=block,
                sae=sae,
                feature_ids=f_ids,
                feature_scales=f_scales,
            )
        return img, curves

    main_img, main_curves = _run_one("main" + f"_{int_cfg.mode}", int_cfg.t_start, int_cfg.t_end)
    early_img, early_curves = _run_one("early" + f"_{int_cfg.mode}", tw_cfg.early_start, tw_cfg.early_end)
    late_img, late_curves = _run_one("late" + f"_{int_cfg.mode}", tw_cfg.late_start, tw_cfg.late_end)

    compare_imgs = []
    if baseline_img is not None:
        compare_imgs.append(("baseline", baseline_img))
    if main_img is not None:
        compare_imgs.append(("main", main_img))
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

    # 导出曲线：每个 block 一份 CSV，包含 baseline/main/early/late
    for block in blocks:
        csv_path = os.path.join(root, f"suite_curve_{safe_name(block)}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step_idx", "timestep", "baseline", "main", "early", "late"])
            steps_main, ts_main, vals_main = main_curves[block]
            steps_base, ts_base, vals_base = baseline_curves.get(block, ([], [], []))
            steps_early, ts_early, vals_early = early_curves.get(block, ([], [], []))
            steps_late, ts_late, vals_late = late_curves.get(block, ([], [], []))
            n = len(steps_main)
            for i in range(n):
                writer.writerow(
                    [
                        steps_main[i],
                        ts_main[i],
                        vals_base[i] if vals_base else "",
                        vals_main[i],
                        vals_early[i] if vals_early else "",
                        vals_late[i] if vals_late else "",
                    ]
                )
        print(f"已保存曲线: {csv_path}")
