"""
实验 55：真实图像概念定位验刀（Noisy Latent Probe）。

严格流程：
1) 从真实图像数据集采样；
2) VAE 编码得到 x0，并加噪到指定时间步 t 得到 xt；
3) 用空文本条件（""）做一次 U-Net 前向；
4) 在指定 block 取 delta = h_out - h_in，送入 SAE；
5) 对目标特征做空间聚合（默认全局 Max Pool），对全数据集排序；
6) 导出 Top-N 图像与空间热力图叠加，检查是否“真概念”而非伪相关。
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

from ..configs import Exp55Config, SAEConfig
from ..core.session import SDXLExperimentSession
from ..utils import block_short_name, ensure_dir, normalize_01, overlay_heatmap, safe_name


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_image_paths(root: str, *, recursive: bool, max_images: int, seed: int) -> List[Path]:
    """扫描数据集目录并随机采样固定数量图像。"""
    p = Path(os.path.expanduser(root)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"exp55 image_root 不存在: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"exp55 image_root 不是目录: {p}")

    if bool(recursive):
        paths = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]
    else:
        paths = [x for x in p.glob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]

    if not paths:
        raise FileNotFoundError(f"exp55 未找到图像文件: {p}")

    rng = random.Random(int(seed))
    rng.shuffle(paths)
    n = int(max_images)
    if n > 0:
        paths = paths[:n]
    return paths


def _build_t_values(*, noise_t: int, t_start: int, t_end: int, num_t_samples: int) -> List[int]:
    """构建要评估的时间步列表。"""
    n = int(num_t_samples)
    if n <= 1:
        return [int(noise_t)]
    lo = int(min(t_start, t_end))
    hi = int(max(t_start, t_end))
    vals = np.linspace(hi, lo, num=n, dtype=int).tolist()
    out: List[int] = []
    seen = set()
    for v in vals:
        x = int(v)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _save_score_distribution_plots(
    *,
    records: List[Dict],
    out_dir: str,
) -> None:
    """保存分数分布图，并绘制多个分位线。"""
    if not records:
        return
    ensure_dir(out_dir)

    scores = np.array([float(r.get("score", 0.0)) for r in records], dtype=np.float64)
    if scores.size == 0:
        return

    qs = [0.50, 0.75, 0.90, 0.95, 0.99]
    qvals = {q: float(np.quantile(scores, q)) for q in qs}
    bins = int(min(80, max(20, int(np.sqrt(float(scores.size)) * 2))))

    plt.figure(figsize=(9, 4.5))
    plt.hist(scores, bins=bins, color="#4c78a8", alpha=0.75, edgecolor="white")
    for q in qs:
        x = qvals[q]
        plt.axvline(x, linestyle="--", linewidth=1.2, label=f"P{int(q*100)}={x:.4g}")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Score Frequency Distribution with Quantile Lines")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_distribution_quantiles.png"), dpi=160)
    plt.close()

    # 频次聚合模式下，再画 hit_count 频率分布
    if any("hit_count" in r for r in records):
        hit_counts = np.array([int(r.get("hit_count", 0)) for r in records], dtype=np.int32)
        if hit_counts.size > 0:
            min_h = int(hit_counts.min())
            max_h = int(hit_counts.max())
            xs = list(range(min_h, max_h + 1))
            ys = [(hit_counts == x).sum() for x in xs]
            plt.figure(figsize=(7.5, 4.2))
            plt.bar(xs, ys, color="#f58518", alpha=0.8, width=0.85)
            plt.xlabel("hit_count")
            plt.ylabel("count")
            plt.title("Hit Count Frequency Distribution")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "hit_count_distribution.png"), dpi=160)
            plt.close()


def _load_blacklist_ids(path: str) -> set[int]:
    """读取黑名单 id（支持 txt/csv）。"""
    ids: set[int] = set()
    p = os.path.expanduser(str(path))
    if not os.path.exists(p):
        return ids
    try:
        if p.endswith(".csv"):
            with open(p, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if r.fieldnames and "feature_id" in r.fieldnames:
                    for row in r:
                        try:
                            ids.add(int(row["feature_id"]))
                        except Exception:
                            continue
                    return ids
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                s = s.split(",")[0].strip()
                try:
                    ids.add(int(s))
                except Exception:
                    continue
    except Exception:
        return ids
    return ids


def _load_image_for_vae(path: Path, *, resolution: int) -> Tuple[Image.Image, torch.Tensor]:
    """读取并预处理图像到 VAE 输入张量 [-1,1]。"""
    img = Image.open(path).convert("RGB")
    r = int(resolution)
    if r > 0:
        img = img.resize((r, r), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [H,W,C], [-1,1]
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return img, ten


def _flatten_spatial(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[str, int, int, int]]:
    """把 block 张量转成 [tokens, d_model]。"""
    if x.dim() == 4:
        b, c, h, w = map(int, x.shape)
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return flat, ("bchw", b, h, w)
    if x.dim() == 3:
        b, n, c = map(int, x.shape)
        flat = x.reshape(b * n, c)
        return flat, ("bnc", b, n, c)
    raise ValueError(f"不支持的张量形状: {tuple(x.shape)}")


def _token_to_heat(token_score: torch.Tensor, meta: Tuple[str, int, int, int]) -> torch.Tensor:
    """把 token 级分数还原为空间热图。"""
    kind, a, b, c = meta
    if kind == "bchw":
        bsz, h, w = int(a), int(b), int(c)
        return token_score.reshape(bsz, h, w)[0].detach().float().cpu()
    if kind == "bnc":
        bsz, n = int(a), int(b)
        side = int(round(float(n) ** 0.5))
        if side > 0 and side * side == n:
            return token_score.reshape(bsz, n)[0].reshape(side, side).detach().float().cpu()
        return token_score.reshape(bsz, n)[0].reshape(1, n).detach().float().cpu()
    raise ValueError(f"未知 meta: {meta}")


def _load_concept_features(
    *,
    block: str,
    concept_name: str,
    top_k: int,
) -> Tuple[List[int], List[float], str]:
    """从 exp53 输出读取目标概念的 top 特征与分数。"""
    concept = str(concept_name).strip()
    if not concept:
        raise ValueError("exp55 需要提供 concept_name。")

    block_tag = block_short_name(block)
    concept_dir = os.path.join(f"out_concept_dict_{block_tag}", concept)
    csv_path = os.path.join(concept_dir, "top_positive_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"exp55 缺少概念特征 CSV: {csv_path}（请先跑 exp53）")

    rows: List[Tuple[int, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fid = int(row["feature_id"])
                score = float(row["score"])
            except Exception:
                continue
            rows.append((fid, score))
    if not rows:
        raise ValueError(f"exp55 概念 CSV 为空: {csv_path}")

    k = int(top_k)
    if k > 0:
        rows = rows[:k]
    feature_ids = [int(fid) for fid, _ in rows]
    raw_scores = torch.tensor([float(s) for _, s in rows], dtype=torch.float32)
    # 用 exp53 的分数作为多特征组合权重；全部非正时回退均匀权重
    w = torch.clamp(raw_scores, min=0.0)
    if float(w.sum().item()) <= 1e-12:
        w = torch.ones_like(w)
    w = w / (w.mean() + 1e-12)  # 均值归一为 1
    feature_weights = [float(x) for x in w.tolist()]
    return feature_ids, feature_weights, concept_dir


def _prepare_empty_condition(
    *,
    pipe,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 SDXL 空文本条件（无 CFG）。"""
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt="",
        prompt_2="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

    proj_dim = None
    if getattr(pipe, "text_encoder_2", None) is not None:
        proj_dim = getattr(getattr(pipe.text_encoder_2, "config", None), "projection_dim", None)
    try:
        add_time_ids = pipe._get_add_time_ids(
            (int(height), int(width)),
            (0, 0),
            (int(height), int(width)),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=proj_dim,
        )
    except TypeError:
        add_time_ids = pipe._get_add_time_ids(
            (int(height), int(width)),
            (0, 0),
            (int(height), int(width)),
            dtype=prompt_embeds.dtype,
        )
    add_time_ids = add_time_ids.to(device=device, dtype=dtype)
    if add_time_ids.dim() == 1:
        add_time_ids = add_time_ids.unsqueeze(0)
    return prompt_embeds, pooled_prompt_embeds, add_time_ids


def _capture_block_delta_once(
    *,
    session: SDXLExperimentSession,
    block: str,
    latents_t: torch.Tensor,
    noise_t: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
) -> torch.Tensor:
    """在给定 xt 与 t 下做一次 U-Net 前向，并抓取 block 的 delta。"""
    cap: Dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        x_in = input[0] if isinstance(input, tuple) else input
        x_out = output[0] if isinstance(output, tuple) else output
        if not isinstance(x_in, torch.Tensor) or not isinstance(x_out, torch.Tensor):
            raise RuntimeError("hook 捕获到非 tensor 输入/输出。")
        cap["in"] = x_in.detach()
        cap["out"] = x_out.detach()

    handle = session.pipe._locate_block(block).register_forward_hook(_hook)
    try:
        unet = session.pipe.pipe.unet
        t = torch.tensor([int(noise_t)], device=latents_t.device, dtype=torch.long)
        _ = unet(
            latents_t.to(device=latents_t.device, dtype=next(unet.parameters()).dtype),
            t,
            encoder_hidden_states=prompt_embeds.to(device=latents_t.device, dtype=next(unet.parameters()).dtype),
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds.to(device=latents_t.device, dtype=next(unet.parameters()).dtype),
                "time_ids": add_time_ids.to(device=latents_t.device, dtype=next(unet.parameters()).dtype),
            },
            return_dict=False,
        )
    finally:
        handle.remove()

    if "in" not in cap or "out" not in cap:
        raise RuntimeError(f"exp55 未捕获到 block={block} 的输入/输出。")
    return cap["out"] - cap["in"]


@torch.no_grad()
def run_exp55_noisy_latent_probe(
    model_cfg,
    sae_cfg: SAEConfig,
    exp55_cfg: Exp55Config,
    output_dir: str,
) -> None:
    """执行 exp55 主流程。"""
    block = str(exp55_cfg.block)
    image_root = str(exp55_cfg.image_root).strip()
    if not image_root:
        raise ValueError("exp55 需要提供 --exp55_image_root。")

    feature_ids, feature_weights, concept_dir = _load_concept_features(
        block=block,
        concept_name=str(exp55_cfg.concept_name),
        top_k=int(exp55_cfg.feature_top_k),
    )

    t_values = _build_t_values(
        noise_t=int(exp55_cfg.noise_t),
        t_start=int(getattr(exp55_cfg, "t_start", exp55_cfg.noise_t)),
        t_end=int(getattr(exp55_cfg, "t_end", exp55_cfg.noise_t)),
        num_t_samples=int(getattr(exp55_cfg, "num_t_samples", 1)),
    )
    t_aggregate = str(getattr(exp55_cfg, "t_aggregate", "freq")).lower()
    if t_aggregate not in {"freq", "mean"}:
        t_aggregate = "freq"

    out_root = os.path.join(
        output_dir,
        f"exp55_{safe_name(block_short_name(block))}_{safe_name(str(exp55_cfg.concept_name))}_t{min(t_values)}-{max(t_values)}_n{len(t_values)}_{t_aggregate}",
    )
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "top_images"))

    session = SDXLExperimentSession(model_cfg, sae_cfg)
    session.load_saes([block])
    sae = session.get_sae(block)
    p_sae = next(sae.parameters())
    pipe = session.pipe.pipe
    device = torch.device(session.device)

    image_paths = _collect_image_paths(
        image_root,
        recursive=bool(exp55_cfg.recursive),
        max_images=int(exp55_cfg.max_images),
        seed=int(exp55_cfg.dataset_seed),
    )
    print(f"[exp55] 数据集采样数: {len(image_paths)}")
    print(f"[exp55] 概念目录: {concept_dir}")
    print(f"[exp55] block={block}, feature_ids={feature_ids}")
    print(f"[exp55] t_values={t_values}, aggregate={t_aggregate}")

    # 预先构建空文本条件（同分辨率共享）
    res = int(exp55_cfg.resolution)
    prompt_embeds, pooled_prompt_embeds, add_time_ids = _prepare_empty_condition(
        pipe=pipe,
        height=res,
        width=res,
        device=device,
        dtype=next(pipe.unet.parameters()).dtype,
    )

    records: List[Dict] = []
    failed = 0
    for img_path in tqdm(image_paths, desc="exp55 正向+打分"):
        try:
            img_pil, img_tensor = _load_image_for_vae(img_path, resolution=res)
            vae_dtype = next(pipe.vae.parameters()).dtype
            img_tensor = img_tensor.to(device=device, dtype=vae_dtype)
            posterior = pipe.vae.encode(img_tensor).latent_dist
            latents_0 = posterior.sample() * float(getattr(pipe.vae.config, "scaling_factor", 0.18215))

            pool_mode = str(exp55_cfg.pooling).lower()
            score_ts: List[float] = []
            feat_sum = None
            heat_sum = None
            meta = None
            noise = torch.randn_like(latents_0)
            for t_int in t_values:
                t = torch.tensor([int(t_int)], device=device, dtype=torch.long)
                latents_t = pipe.scheduler.add_noise(latents_0, noise, t)

                delta = _capture_block_delta_once(
                    session=session,
                    block=block,
                    latents_t=latents_t,
                    noise_t=int(t_int),
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    add_time_ids=add_time_ids,
                )

                flat, meta = _flatten_spatial(delta.to(device=p_sae.device, dtype=p_sae.dtype))
                z = sae.encode(flat)  # [tokens, n_features]
                ids_t = torch.tensor(feature_ids, device=z.device, dtype=torch.long)
                sel = torch.relu(z[:, ids_t])  # [tokens, k]
                w = torch.tensor(feature_weights, device=sel.device, dtype=sel.dtype)  # [k]
                token_score = sel @ w  # [tokens]

                if pool_mode == "topk_mean":
                    k = min(max(1, int(exp55_cfg.pool_topk)), int(token_score.numel()))
                    score_t = float(torch.topk(token_score, k=k).values.mean().item())
                else:
                    score_t = float(token_score.max().item())  # 默认严格用全局 max pooling
                score_ts.append(score_t)

                feat_t = sel.max(dim=0).values.detach().float().cpu()
                heat_t = _token_to_heat(token_score, meta).detach().float().cpu()
                feat_sum = feat_t if feat_sum is None else (feat_sum + feat_t)
                heat_sum = heat_t if heat_sum is None else (heat_sum + heat_t)

            score_mean = float(sum(score_ts) / max(1, len(score_ts)))
            feat_max = feat_sum / float(max(1, len(t_values)))
            heat = heat_sum / float(max(1, len(t_values)))
            records.append(
                {
                    "path": str(img_path),
                    "score": score_mean,
                    "score_ts": score_ts,
                    "feature_max": feat_max,
                    "heat": heat,
                    "image_size": img_pil.size,
                }
            )
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            failed += 1
            print(f"[exp55] 跳过图像: {img_path} | err={e}")
            continue

    if not records:
        raise RuntimeError("exp55 没有得到任何有效样本，请检查数据集和参数。")

    # 多 t 聚合：
    # - mean: 按 score 均值排序
    # - freq: 统计每个 t 的 Top-N 出现频次，再按频次/命中分数排序
    if t_aggregate == "freq" and len(t_values) > 1:
        top_n_for_freq = min(int(exp55_cfg.top_n), len(records))
        for r in records:
            r["hit_count"] = 0
            r["hit_rank_sum"] = 0.0
            r["hit_score_sum"] = 0.0
        for ti, t_int in enumerate(t_values):
            order = sorted(range(len(records)), key=lambda idx: float(records[idx]["score_ts"][ti]), reverse=True)
            top_ids = order[:top_n_for_freq]
            for rk, idx in enumerate(top_ids, start=1):
                rec = records[idx]
                rec["hit_count"] += 1
                rec["hit_rank_sum"] += float(rk)
                rec["hit_score_sum"] += float(rec["score_ts"][ti])
        for r in records:
            hc = int(r["hit_count"])
            r["hit_rank_mean"] = float(r["hit_rank_sum"] / hc) if hc > 0 else 1e9
            r["hit_score_mean"] = float(r["hit_score_sum"] / hc) if hc > 0 else 0.0
        records.sort(
            key=lambda x: (
                -int(x["hit_count"]),
                -float(x["hit_score_mean"]),
                float(x["hit_rank_mean"]),
                -float(x["score"]),
            )
        )
    else:
        records.sort(key=lambda x: float(x["score"]), reverse=True)
    top_n = min(int(exp55_cfg.top_n), len(records))

    # 全量排序结果
    all_csv = os.path.join(out_root, "all_scores.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "image_path",
                "score",
                "score_ts",
                "feature_ids",
                "feature_max",
                "hit_count",
                "hit_rank_mean",
                "hit_score_mean",
            ],
        )
        w.writeheader()
        for i, row in enumerate(records, start=1):
            feat_max_str = " ".join(f"{float(x):.6g}" for x in row["feature_max"].tolist())
            score_ts_str = " ".join(f"{float(x):.6g}" for x in row.get("score_ts", []))
            w.writerow(
                {
                    "rank": int(i),
                    "image_path": row["path"],
                    "score": float(row["score"]),
                    "score_ts": score_ts_str,
                    "feature_ids": " ".join(str(int(x)) for x in feature_ids),
                    "feature_max": feat_max_str,
                    "hit_count": int(row.get("hit_count", 0)),
                    "hit_rank_mean": float(row.get("hit_rank_mean", 0.0)),
                    "hit_score_mean": float(row.get("hit_score_mean", 0.0)),
                }
            )

    # 导出每个 t 的 Top-N（便于检查“频次”来源）
    if len(t_values) > 1:
        per_t_csv = os.path.join(out_root, "top_per_t.csv")
        with open(per_t_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "rank", "image_path", "score_t"])
            w.writeheader()
            for ti, t_int in enumerate(t_values):
                order = sorted(range(len(records)), key=lambda idx: float(records[idx]["score_ts"][ti]), reverse=True)
                for rk, idx in enumerate(order[:top_n], start=1):
                    rec = records[idx]
                    w.writerow(
                        {
                            "t": int(t_int),
                            "rank": int(rk),
                            "image_path": rec["path"],
                            "score_t": float(rec["score_ts"][ti]),
                        }
                    )

    # 统计各候选特征在数据集上的激活频率，并给出黑名单建议
    feat_mat = torch.stack([r["feature_max"] for r in records], dim=0)  # [N, K]
    eps = float(getattr(exp55_cfg, "blacklist_eps", 1e-6))
    thr = float(getattr(exp55_cfg, "blacklist_freq_threshold", 0.99))
    active_ratio = (feat_mat > eps).float().mean(dim=0)  # [K]
    mean_act = feat_mat.mean(dim=0)  # [K]
    std_act = feat_mat.std(dim=0, unbiased=False)  # [K]

    freq_csv = os.path.join(out_root, "feature_activity_frequency.csv")
    with open(freq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature_id", "active_ratio", "mean_activation", "std_activation", "blacklisted"],
        )
        w.writeheader()
        for j, fid in enumerate(feature_ids):
            ratio = float(active_ratio[j].item())
            is_bl = int(ratio >= thr)
            w.writerow(
                {
                    "feature_id": int(fid),
                    "active_ratio": ratio,
                    "mean_activation": float(mean_act[j].item()),
                    "std_activation": float(std_act[j].item()),
                    "blacklisted": is_bl,
                }
            )

    auto_blacklist_ids = [int(feature_ids[j]) for j in range(len(feature_ids)) if float(active_ratio[j].item()) >= thr]
    auto_bl_txt = os.path.join(out_root, "feature_blacklist_auto.txt")
    with open(auto_bl_txt, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by exp55\n")
        f.write(f"# rule: active_ratio >= {thr}, eps={eps}\n")
        for fid in sorted(auto_blacklist_ids):
            f.write(f"{int(fid)}\n")

    if bool(getattr(exp55_cfg, "write_blacklist", True)):
        bl_name = str(getattr(exp55_cfg, "blacklist_filename", "feature_blacklist.txt")).strip() or "feature_blacklist.txt"
        concept_bl_path = os.path.join(concept_dir, bl_name)
        prev_ids = _load_blacklist_ids(concept_bl_path)
        merged = sorted(set(prev_ids).union(set(auto_blacklist_ids)))
        ensure_dir(os.path.dirname(concept_bl_path) or ".")
        with open(concept_bl_path, "w", encoding="utf-8") as f:
            f.write("# feature blacklist for concept-level filtering\n")
            f.write(f"# generated_by=exp55\n")
            f.write(f"# threshold={thr}, eps={eps}\n")
            for fid in merged:
                f.write(f"{int(fid)}\n")
        print(f"[exp55] 黑名单写入: {concept_bl_path} (prev={len(prev_ids)} + auto={len(auto_blacklist_ids)} -> merged={len(merged)})")

    # 统计图：分数频率分布 + 分位线；freq 模式额外输出 hit_count 分布
    _save_score_distribution_plots(records=records, out_dir=out_root)

    # Top-N 图像与热力图叠加
    top_csv = os.path.join(out_root, "top_scores.csv")
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "image_path", "score", "hit_count", "score_ts"])
        w.writeheader()
        for rank, row in enumerate(records[:top_n], start=1):
            img = Image.open(row["path"]).convert("RGB").resize((res, res), Image.BICUBIC)
            heat = normalize_01(row["heat"])
            stem = safe_name(Path(row["path"]).stem)
            prefix = f"rank_{rank:03d}_score_{float(row['score']):.6f}_{stem}"

            img_path = os.path.join(out_root, "top_images", f"{prefix}_image.png")
            ovl_path = os.path.join(out_root, "top_images", f"{prefix}_overlay.png")
            npy_path = os.path.join(out_root, "top_images", f"{prefix}_heat.pt")

            img.save(img_path)
            overlay_heatmap(
                heat,
                out_path=ovl_path,
                title=f"rank={rank} score={float(row['score']):.4f}",
                base_image=img,
                alpha=float(exp55_cfg.overlay_alpha),
            )
            torch.save(heat, npy_path)
            w.writerow(
                {
                    "rank": int(rank),
                    "image_path": row["path"],
                    "score": float(row["score"]),
                    "hit_count": int(row.get("hit_count", 0)),
                    "score_ts": " ".join(f"{float(x):.6g}" for x in row.get("score_ts", [])),
                }
            )

    # 元信息
    meta_path = os.path.join(out_root, "run_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"image_root={str(Path(image_root).resolve())}\n")
        f.write(f"num_images_requested={int(exp55_cfg.max_images)}\n")
        f.write(f"num_images_used={len(records)}\n")
        f.write(f"num_images_failed={int(failed)}\n")
        f.write(f"block={block}\n")
        f.write(f"concept_name={str(exp55_cfg.concept_name)}\n")
        f.write(f"feature_ids={' '.join(str(int(x)) for x in feature_ids)}\n")
        f.write(f"feature_weights={' '.join(f'{float(x):.6g}' for x in feature_weights)}\n")
        f.write(f"blacklist_eps={eps}\n")
        f.write(f"blacklist_freq_threshold={thr}\n")
        f.write(f"auto_blacklist_ids={' '.join(str(int(x)) for x in auto_blacklist_ids)}\n")
        f.write(f"noise_t={int(exp55_cfg.noise_t)}\n")
        f.write(f"t_start={int(getattr(exp55_cfg, 't_start', exp55_cfg.noise_t))}\n")
        f.write(f"t_end={int(getattr(exp55_cfg, 't_end', exp55_cfg.noise_t))}\n")
        f.write(f"num_t_samples={int(getattr(exp55_cfg, 'num_t_samples', 1))}\n")
        f.write(f"t_values={' '.join(str(int(x)) for x in t_values)}\n")
        f.write(f"t_aggregate={t_aggregate}\n")
        f.write(f"pooling={str(exp55_cfg.pooling)}\n")
        f.write(f"pool_topk={int(exp55_cfg.pool_topk)}\n")
        f.write(f"resolution={int(exp55_cfg.resolution)}\n")

    print(f"[exp55] 输出目录: {out_root}")
    print(f"[exp55] all_scores: {all_csv}")
    print(f"[exp55] top_scores: {top_csv}")
    print(f"[exp55] top_images: {os.path.join(out_root, 'top_images')}")
