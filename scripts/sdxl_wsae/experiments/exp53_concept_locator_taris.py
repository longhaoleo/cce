"""
实验 53：概念定位（TARIS, Time-Averaged Relative Importance Score）。

目标：
- 给定“概念正样本 prompts 集合” D_c 与“非概念 prompts 集合” D_~c
- 在扩散轨迹的一个时间窗口内，对每个特征 i 计算“时域平均相对重要性分数”
- 输出最稳态的 Top 特征索引（更鲁棒，不容易被单步抖动误导）

核心定义（对应你给的公式，离散化实现）：
GlobalScore(i) = (1/|T|) * sum_{t in T} [ mu_c(i,t)/(E_c(t)+δ) - mu_nc(i,t)/(E_nc(t)+δ) ]
其中：
- mu(i,t,D) 表示在数据集 D 上、时间步 t 的该特征平均激活（这里用 token 维均值 + prompt 维均值）
- E(t,D) = sum_j mu(j,t,D) 是该时刻总能量（归一化分母）

实现约定：
- 我们仍然使用 delta = h_out - h_in 作为 SAE 的输入（更像“该模块在该 step 做了什么更新”）
- 每一步得到 z = sae.encode(delta_tokens)，再对 tokens 求均值得到长度为 n_features 的向量 mu_t
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..configs import ConceptLocateConfig, RunConfig, SAEConfig
from ..core.session import SDXLExperimentSession
from ..utils import ensure_dir, safe_name
from .shared_prepare import DeltaExtractor


@torch.no_grad()
def _c_mat_for_prompt(
    *,
    session: SDXLExperimentSession,
    sae: torch.nn.Module,
    block: str,
    run_cfg: RunConfig,
) -> Tuple[torch.Tensor, List[int]]:
    """对单个 prompt 跑一次轨迹，返回 [steps, n_features] 的激活矩阵与 timesteps 列表。"""
    output, cache = session.run_with_cache(
        run_cfg,
        positions_to_cache=[block],
        save_input=True,
        save_output=True,
        output_type="pil",
    )
    timesteps = session.scheduler_timesteps(session.pipe)
    extractor = DeltaExtractor()
    deltas = extractor.extract(block=block, cache=cache, timesteps=timesteps)

    p = next(sae.parameters())
    rows: List[torch.Tensor] = []
    for item in deltas:
        x = item.x.to(device=p.device, dtype=p.dtype)
        z = sae.encode(x)  # [tokens, n_features]
        rows.append(z.mean(dim=0).detach().float().cpu())  # [n_features]
    return torch.stack(rows, dim=0), timesteps


def _select_step_indices(
    timesteps: Sequence[int],
    *,
    t_start: int,
    t_end: int,
    num_t_samples: int,
) -> List[int]:
    """从 scheduler timesteps 中选出窗口内的 step 索引，并在窗口内均匀采样。"""
    low = int(min(t_start, t_end))
    high = int(max(t_start, t_end))
    candidates = [i for i, t in enumerate(timesteps) if low <= int(t) <= high]
    if not candidates:
        candidates = list(range(len(timesteps)))

    n = int(num_t_samples)
    if n <= 0 or len(candidates) <= n:
        return candidates

    pick_pos = np.linspace(0, len(candidates) - 1, num=n, dtype=int).tolist()
    picked = [candidates[i] for i in pick_pos]
    # 去重（linspace 可能产生重复索引），并保持顺序
    out: List[int] = []
    seen = set()
    for idx in picked:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _taris_score(
    *,
    pos_mu: torch.Tensor,  # [steps, n_features]
    neg_mu: torch.Tensor,  # [steps, n_features]
    step_indices: Sequence[int],
    delta: float,
) -> torch.Tensor:
    """按 TARIS 公式对选定 step 做时间平均，返回 [n_features] 分数向量。"""
    if pos_mu.shape != neg_mu.shape:
        raise ValueError(f"pos/neg 形状不一致: {tuple(pos_mu.shape)} vs {tuple(neg_mu.shape)}")
    if not step_indices:
        raise ValueError("step_indices 为空，无法计算 TARIS。")

    d = float(delta)
    scores = torch.zeros(int(pos_mu.shape[1]), dtype=torch.float32)
    for idx in step_indices:
        mu_c = pos_mu[int(idx)]
        mu_nc = neg_mu[int(idx)]
        e_c = float(mu_c.sum().item()) + d
        e_nc = float(mu_nc.sum().item()) + d
        score_t = (mu_c / e_c) - (mu_nc / e_nc)
        scores += score_t
    return scores / float(len(step_indices))


def _save_topk_csv(path: str, *, top_ids: torch.Tensor, top_vals: torch.Tensor) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "feature_id", "score"])
        w.writeheader()
        for r, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            w.writerow({"rank": int(r), "feature_id": int(fid), "score": float(val)})


def run_exp53_concept_locator_taris(
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    concept_cfg: ConceptLocateConfig,
    output_dir: str,
    *,
    session: SDXLExperimentSession | None = None,
) -> None:
    """运行实验 53：TARIS 概念定位。"""
    ensure_dir(output_dir)

    block = str(concept_cfg.block)
    pos_prompts = list(concept_cfg.pos_prompts)
    neg_prompts = list(concept_cfg.neg_prompts)
    if not pos_prompts or not neg_prompts:
        raise ValueError("exp53 需要同时提供 --pos_prompts 和 --neg_prompts（至少各 1 条）。")

    # 1) session 允许外部传入（批量跑多个概念时避免重复加载 SDXL/SAE）。
    local_session = session if session is not None else SDXLExperimentSession(model_cfg, sae_cfg)
    local_session.load_saes([block])
    sae = local_session.get_sae(block)

    # 2) 计算每个 prompt 的 [steps, n_features]，再在 prompt 维度求均值，得到 mu(t, D)
    pos_acc = None
    pos_timesteps: List[int] = []
    for ptxt in pos_prompts:
        cfg = replace(run_cfg, prompt=str(ptxt))
        c_mat, ts = _c_mat_for_prompt(session=local_session, sae=sae, block=block, run_cfg=cfg)
        if pos_acc is None:
            pos_acc = c_mat
            pos_timesteps = list(ts)
        else:
            if list(ts) != pos_timesteps:
                raise ValueError("不同 prompt 的 scheduler timesteps 不一致，无法对齐做时域积分。")
            pos_acc = pos_acc + c_mat
    assert pos_acc is not None
    pos_mu = pos_acc / float(len(pos_prompts))

    neg_acc = None
    neg_timesteps: List[int] = []
    for ptxt in neg_prompts:
        cfg = replace(run_cfg, prompt=str(ptxt))
        c_mat, ts = _c_mat_for_prompt(session=local_session, sae=sae, block=block, run_cfg=cfg)
        if neg_acc is None:
            neg_acc = c_mat
            neg_timesteps = list(ts)
        else:
            if list(ts) != neg_timesteps:
                raise ValueError("不同 prompt 的 scheduler timesteps 不一致，无法对齐做时域积分。")
            neg_acc = neg_acc + c_mat
    assert neg_acc is not None
    neg_mu = neg_acc / float(len(neg_prompts))

    if pos_timesteps != neg_timesteps:
        raise ValueError("正负 prompt 的 scheduler timesteps 不一致，请确保 steps 等采样参数一致。")

    # 3) 选择时间窗口内的 step，然后做 TARIS 积分/平均
    step_indices = _select_step_indices(
        pos_timesteps,
        t_start=int(concept_cfg.t_start),
        t_end=int(concept_cfg.t_end),
        num_t_samples=int(concept_cfg.num_t_samples),
    )
    scores = _taris_score(
        pos_mu=pos_mu,
        neg_mu=neg_mu,
        step_indices=step_indices,
        delta=float(concept_cfg.delta),
    )

    # 4) 输出 Top-K（正向/反向都给一份，便于看“反概念特征”）
    k = max(1, int(concept_cfg.top_k))
    top_pos_vals, top_pos_ids = torch.topk(scores, k=min(k, int(scores.numel())))
    # 说明：这里仅输出“正向端”Top-K（scores 最大的一侧）。

    # 输出目录组织：
    # - 传了 concept_name：output_dir/exp53_taris/{concept_name}/
    # - 不传：             output_dir/exp53_taris/
    concept_name = safe_name(str(concept_cfg.concept_name or "").strip())
    if concept_name:
        out_dir = os.path.join(output_dir, "exp53_taris", concept_name)
    else:
        out_dir = os.path.join(output_dir, "exp53_taris")
    ensure_dir(out_dir)

    meta_path = os.path.join(out_dir, "taris_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"block={block}\n")
        f.write(f"pos_prompts={pos_prompts}\n")
        f.write(f"neg_prompts={neg_prompts}\n")
        f.write(f"t_start={int(concept_cfg.t_start)}\n")
        f.write(f"t_end={int(concept_cfg.t_end)}\n")
        f.write(f"num_t_samples={int(concept_cfg.num_t_samples)}\n")
        f.write(f"delta={float(concept_cfg.delta)}\n")
        f.write(f"selected_step_indices={list(map(int, step_indices))}\n")
        f.write(f"selected_timesteps={[int(pos_timesteps[i]) for i in step_indices]}\n")

    _save_topk_csv(
        os.path.join(out_dir, "taris_top_positive.csv"),
        top_ids=top_pos_ids,
        top_vals=top_pos_vals,
    )

    # 保存一份 tensor 包，后续你可以直接加载做更多统计/可视化
    torch.save(
        {
            "block": block,
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts,
            "timesteps": pos_timesteps,
            "selected_step_indices": list(map(int, step_indices)),
            "scores": scores,
            "top_positive_ids": top_pos_ids,
            "top_positive_vals": top_pos_vals,
            "pos_mu": pos_mu,
            "neg_mu": neg_mu,
        },
        os.path.join(out_dir, "taris_dump.pt"),
    )

    # 5) 分布可视化：所有特征的 TARIS 得分直方图（柱状图）
    # 注意：特征维通常是几万，逐个特征画 bar 会不可读；直方图更适合看整体分布与尾部厚度。
    plt.figure(figsize=(10, 4))
    s_np = scores.detach().float().cpu().numpy()
    plt.hist(s_np, bins=200, color="#4C72B0", alpha=0.9)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("TARIS score")
    plt.ylabel("feature count")
    plt.title("TARIS Score Distribution (All Features)")
    # feature count 往往跨度很大，用对数刻度更容易看“只有少数很大”的长尾结构
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "taris_scores_hist.png"), dpi=150)
    plt.close()

    # 6) “只有少数比较大吗？”最直观的两张图：
    # - Rank 曲线：把得分按大小排序，画 score(rank)，尾部是否陡峭一眼就能看出来
    # - 累积贡献：Top-x% 的特征累积贡献了多少“正向得分质量”(mass)
    s = scores.detach().float().cpu().numpy().astype(np.float64)
    s_sorted = np.sort(s)[::-1]  # 从大到小

    # Rank plot：rank 维度非常长，用对数 x 轴更容易同时看清头部与长尾
    plt.figure(figsize=(10, 4))
    ranks = np.arange(1, len(s_sorted) + 1, dtype=np.int64)
    plt.plot(ranks, s_sorted, linewidth=1.0)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xscale("log")
    plt.xlabel("feature rank (log scale, sorted by TARIS score desc)")
    plt.ylabel("TARIS score")
    plt.title("TARIS Scores Rank Plot (Descending, log-rank)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "taris_scores_rank.png"), dpi=150)
    plt.close()

    # Cumulative mass on positive scores
    pos = s_sorted[s_sorted > 0]
    if pos.size > 0:
        cum = np.cumsum(pos)
        total = float(cum[-1])
        frac_feat = (np.arange(1, pos.size + 1) / float(pos.size))
        frac_mass = cum / max(total, 1e-12)

        plt.figure(figsize=(10, 4))
        plt.plot(frac_feat, frac_mass, linewidth=2.0)
        plt.xlabel("fraction of positive-scoring features (top -> tail)")
        plt.ylabel("fraction of total positive score mass")
        plt.title("TARIS Positive Mass Concentration (Cumulative)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "taris_positive_mass_cumulative.png"), dpi=150)
        plt.close()

        # 打印几个常用分位点，帮助你快速判断“少数特征是否占大头”
        for pct in (0.01, 0.05, 0.10):
            k_feat = max(1, int(round(pct * pos.size)))
            mass_pct = float(cum[k_feat - 1] / max(total, 1e-12))
            print(f"[exp53] Top {int(pct*100)}% positive features capture {mass_pct*100:.2f}% positive mass")

    # 5) 画一个简单曲线图：Top-5 特征在正/负集合上随 step 的平均激活（便于检查“稳态”）
    curve_k = min(5, int(top_pos_ids.numel()))
    if curve_k > 0:
        plt.figure(figsize=(10, 4))
        xs = list(range(int(pos_mu.shape[0])))
        for j in range(curve_k):
            fid = int(top_pos_ids[j].item())
            plt.plot(xs, pos_mu[:, fid].numpy(), label=f"pos f{fid}")
            plt.plot(xs, neg_mu[:, fid].numpy(), linestyle="--", label=f"neg f{fid}")
        plt.xlabel("step_idx")
        plt.ylabel("mean activation")
        plt.title("TARIS Top Features Activation Curves (pos solid / neg dashed)")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "taris_top_curves.png"), dpi=150)
        plt.close()

    print(f"[exp53] 输出目录: {out_dir}")
    print(f"[exp53] Top+ (scores 最大) feature ids: {top_pos_ids[:10].tolist()}")
