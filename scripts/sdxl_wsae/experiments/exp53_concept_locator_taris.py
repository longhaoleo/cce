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
import json
import os
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from ..configs import ConceptLocateConfig, RunConfig, SAEConfig
from ..core.session import SDXLExperimentSession
from ..utils import ensure_dir, safe_name, block_short_name
from .shared_prepare import DeltaExtractor


def _load_concept_prompts_from_json(*, concept_name: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """从 `target_concept_dict/{concept_name}.json` 读取正/负 prompts。

    你这边的 json 格式可能会不断演化，所以这里做成“尽量鲁棒”的解析器：
    - 允许字段名变化：pos/neg/positive/negative/pos_prompts/neg_prompts/...
    - 允许嵌套结构：{"pos": {"prompts": [...]}}、{"positive": [{"text": "..."}]} 等
    - 允许单个字符串：{"pos_prompts": "xxx"}（会转成长度 1 的 list）

    约定：
    - 解析出的 prompts 会做 strip，并过滤空字符串。
    - 最终必须同时得到非空的 pos_prompts 与 neg_prompts。
    """
    name = str(concept_name).strip()
    if not name:
        raise ValueError("exp53 需要提供 --concept_name，用它定位 json 文件名。")

    root = Path("target_concept_dict")
    path = (root / f"{name}.json").resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到概念 json: {path}（请放在 target_concept_dict/ 下，文件名=concept_name.json）")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"概念 json 必须是 dict: {path}")

    def _as_list(x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            x = x.strip()
            return [x] if x else []
        if isinstance(x, list):
            out = []
            for it in x:
                if it is None:
                    continue
                if isinstance(it, str):
                    s = it.strip()
                    if s:
                        out.append(s)
                    continue
                if isinstance(it, dict):
                    # 常见：{"text": "..."} / {"prompt": "..."} / {"caption": "..."}
                    for key in ("prompt", "text", "caption", "value"):
                        if key in it:
                            out.extend(_as_list(it.get(key)))
                            break
                    continue
                s = str(it).strip()
                if s:
                    out.append(s)
            return out
        if isinstance(x, dict):
            # 常见：{"prompts": [...]} 或更深层嵌套
            for key in ("prompts", "prompt", "texts", "text", "captions"):
                if key in x:
                    return _as_list(x.get(key))
            # 不认识的 dict：把所有 value 扫一遍（例如 {"easy":[...], "hard":[...]}）
            out: List[str] = []
            for v in x.values():
                out.extend(_as_list(v))
            return out
        raise ValueError(f"prompts 字段必须是 string 或 list: got {type(x)}")

    POS_KEYS = (
        "pos_prompts",
        "pos",
        "positive_prompts",
        "positive",
        "concept_prompts",
        "concept",
        "target_prompts",
        "target",
    )
    NEG_KEYS = (
        "neg_prompts",
        "neg",
        "negative_prompts",
        "negative",
        "non_concept_prompts",
        "non_concept",
        "control_prompts",
        "control",
        "anti_prompts",
        "anti",
    )

    def _pick_first(keys: Tuple[str, ...]) -> Any:
        for k in keys:
            if k in data:
                return data.get(k)
        return None

    pos = _as_list(_pick_first(POS_KEYS))
    neg = _as_list(_pick_first(NEG_KEYS))
    return pos, neg, data


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
    """保存 Top-K 特征列表（最小格式）。"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "feature_id", "score"])
        w.writeheader()
        for r, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            w.writerow({"rank": int(r), "feature_id": int(fid), "score": float(val)})


def _save_feature_time_scores_csv(
    path: str,
    *,
    timesteps: Sequence[int],
    feature_ids: Sequence[int],
    scores: torch.Tensor,  # [n_features]
    pos_mu: torch.Tensor,  # [steps, n_features]
    neg_mu: torch.Tensor,  # [steps, n_features]
) -> None:
    """保存“指定特征集合”的按 step 激活曲线（用于 exp54 的更细粒度控制）。

    CSV（长表）字段：
    - step_idx, timestep, feature_id
    - taris_score: 该特征的全局 TARIS 分数（与 time 无关）
    - pos_mu, neg_mu, diff
    """
    ensure_dir(os.path.dirname(path) or ".")
    ids = [int(x) for x in feature_ids]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "timestep",
                "feature_id",
                "taris_score",
                "pos_mu",
                "neg_mu",
                "diff",
            ],
        )
        w.writeheader()

        for step_idx, t in enumerate(list(timesteps)):
            for fid in ids:
                w.writerow(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(t),
                        "feature_id": int(fid),
                        "taris_score": float(scores[int(fid)].item()),
                        "pos_mu": float(pos_mu[int(step_idx), int(fid)].item()),
                        "neg_mu": float(neg_mu[int(step_idx), int(fid)].item()),
                        "diff": float((pos_mu[int(step_idx), int(fid)] - neg_mu[int(step_idx), int(fid)]).item()),
                    }
                )


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
    block = str(concept_cfg.block)

    # 0) prompts 输入：统一从 `target_concept_dict/{concept_name}.json` 读取。
    #    这样你可以很方便地扩展 prompt 数量，而不需要在命令行写很长的列表。
    concept_name_raw = str(concept_cfg.concept_name or "").strip()
    pos_prompts, neg_prompts, raw_json = _load_concept_prompts_from_json(concept_name=concept_name_raw)
    if not pos_prompts or not neg_prompts:
        raise ValueError(f"概念 json 的 pos/neg prompts 不能为空: concept={concept_name_raw}")

    # 1) 输出目录固定组织为 out_concept_dict_<block_short>/{concept_name}/
    concept_name = safe_name(concept_name_raw)
    block_tag = block_short_name(str(block))
    out_dir = os.path.join(f"out_concept_dict_{block_tag}", concept_name)
    ensure_dir(out_dir)

    # 2) session 允许外部传入（批量跑多个概念时避免重复加载 SDXL/SAE）。
    local_session = session if session is not None else SDXLExperimentSession(model_cfg, sae_cfg)
    local_session.load_saes([block])
    sae = local_session.get_sae(block)

    # 3) 计算每个 prompt 的 [steps, n_features]，再在 prompt 维度求均值，得到 mu(t, D)
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

    # 4) 选择时间窗口内的 step，然后做 TARIS 积分/平均
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

    # 5) 输出 Top-K（只保留“正向端”，即 scores 最大的那一侧）
    k = max(1, int(concept_cfg.top_k))
    top_pos_vals, top_pos_ids = torch.topk(scores, k=min(k, int(scores.numel())))
    # 负向 Top-K：scores 最小（反概念端），暂时保留，后续你可以用来做“anti-concept”分析
    top_neg_vals, top_neg_ids = torch.topk(scores, k=min(k, int(scores.numel())), largest=False)

    meta_path = os.path.join(out_dir, "taris_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"block={block}\n")
        f.write(f"concept_name={concept_name_raw}\n")
        f.write(f"concept_json={os.path.join('target_concept_dict', concept_name_raw + '.json')}\n")
        f.write(f"pos_prompts={pos_prompts}\n")
        f.write(f"neg_prompts={neg_prompts}\n")
        f.write(f"t_start={int(concept_cfg.t_start)}\n")
        f.write(f"t_end={int(concept_cfg.t_end)}\n")
        f.write(f"num_t_samples={int(concept_cfg.num_t_samples)}\n")
        f.write(f"delta={float(concept_cfg.delta)}\n")
        f.write(f"selected_step_indices={list(map(int, step_indices))}\n")
        f.write(f"selected_timesteps={[int(pos_timesteps[i]) for i in step_indices]}\n")
        # 原始 json 的其他字段也写一下，方便追溯（例如你未来可能加注释/标签）
        extras = {k: v for k, v in raw_json.items() if k not in {"pos_prompts", "neg_prompts", "pos", "neg"}}
        if extras:
            f.write(f"extras={extras}\n")


    # 保存 CSV（正向/负向都输出一份，便于检查与后续扩展）
    _save_topk_csv(os.path.join(out_dir, "top_positive_features.csv"), top_ids=top_pos_ids, top_vals=top_pos_vals)
    _save_topk_csv(os.path.join(out_dir, "top_negative_features.csv"), top_ids=top_neg_ids, top_vals=top_neg_vals)

    # 额外输出：Top+ 特征在不同 t/step 的平均激活（pos/neg/diff）
    # 用于 exp54 做更细粒度的时间调制（不必每次推理都从 x 重新估计系数）。
    _save_feature_time_scores_csv(
        os.path.join(out_dir, "feature_time_scores.csv"),
        timesteps=pos_timesteps,
        feature_ids=top_pos_ids.tolist(),
        scores=scores,
        pos_mu=pos_mu,
        neg_mu=neg_mu,
    )

    # 保存一份 tensor 包，后续你可以直接加载做更多统计/可视化
    torch.save(
        {
            "block": block,
            "concept_name": concept_name_raw,
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts,
            "timesteps": pos_timesteps,
            "selected_step_indices": list(map(int, step_indices)),
            "scores": scores,
            "top_positive_ids": top_pos_ids,
            "top_positive_vals": top_pos_vals,
            "top_negative_ids": top_neg_ids,
            "top_negative_vals": top_neg_vals,
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


    s = scores.detach().float().cpu().numpy().astype(np.float64)
    s_sorted = np.sort(s)[::-1]  # 从大到小

    # 画一个简单曲线图：Top-5 特征在正/负集合上随 step 的平均激活（便于检查“稳态”）
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



    # ==========================================
    #      可视化升级：让结果“看得见、摸得着”
    # ==========================================

    # Visualization 1: 蝴蝶图 (Diverging Bar Chart)
    # 直观展示：红色代表概念端，蓝色代表反概念端
    plt.figure(figsize=(10, 6))

    disp_k = min(20, int(top_pos_ids.numel()), int(top_neg_ids.numel()))
    pos_v = top_pos_vals[:disp_k].cpu().numpy()[::-1]  # 倒序：最大的在最上
    neg_v = top_neg_vals[:disp_k].cpu().numpy()[::-1]

    y_pos = np.arange(len(pos_v))
    plt.barh(y_pos, pos_v, color="#d62728", label="Positive (Concept)", alpha=0.85)
    plt.barh(y_pos, neg_v, color="#1f77b4", label="Negative (Anti-Concept)", alpha=0.85)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.7)
    plt.title(f"Top TARIS Scores: {concept_name}")
    plt.xlabel("TARIS Score (Relative Importance Diff)")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_1_butterfly_chart.png"), dpi=150)
    plt.close()

    # Visualization 2: 差分动力学曲线 (Differential Dynamics)
    # 展示 Top-5 特征的 (Pos - Neg) 随时间变化，看它们在哪个时间段“发力”
    plt.figure(figsize=(12, 5))
    xs = list(range(int(pos_mu.shape[0])))
    
    # 只画 Top-5 Positive
    viz_k = 5
    colors = plt.cm.Reds(np.linspace(0.5, 1, viz_k))
    
    for i in range(viz_k):
        fid = int(top_pos_ids[i].item())
        # 计算差值：这就是 TARIS 的分子部分随时间的变化
        # 归一化一下便于在同一张图显示趋势
        diff_curve = (pos_mu[:, fid] - neg_mu[:, fid]).cpu().numpy()
        
        plt.plot(xs, diff_curve, color=colors[viz_k-1-i], linewidth=2, label=f"Feat {fid}")
        
        # 标出峰值位置
        peak_idx = np.argmax(diff_curve)
        plt.scatter(peak_idx, diff_curve[peak_idx], color=colors[viz_k-1-i], s=30)

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Timesteps (Generation Steps)")
    plt.ylabel("Activity Difference (Pos - Neg)")
    plt.title(f"When do the Top Features act? (Time Dynamics)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_2_diff_dynamics.png"), dpi=150)
    plt.close()

    # Visualization 3: 特征流热力图 (The Waterfall / Heatmap) —— 最核心的图！
    # X轴：时间，Y轴：Top-50 特征（按峰值时间排序）
    # 这张图能证明你的 ODE 流动理论
    
    heatmap_k = 50
    target_ids = top_pos_ids[:heatmap_k].cpu()
    
    # 提取这些特征在 Positive Prompt 下的激活矩阵 [Steps, K]
    # 注意：这里我们看 Pos 激活，因为我们关注概念是如何生成的
    act_matrix = pos_mu[:, target_ids].cpu().numpy().T  # 转置为 [K, Steps]
    
    # 为了热力图好看，对每个特征做 Min-Max 归一化，这样无论强弱都能看到“亮起”的时间
    row_mins = act_matrix.min(axis=1, keepdims=True)
    row_maxs = act_matrix.max(axis=1, keepdims=True) + 1e-6
    act_norm = (act_matrix - row_mins) / (row_maxs - row_mins)
    
    # 关键步骤：按“峰值出现时间”对特征进行排序
    peak_times = np.argmax(act_norm, axis=1) # 找到每个特征最亮的时间点
    sorted_indices = np.argsort(peak_times)  # 按时间排序
    act_sorted = act_norm[sorted_indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(act_sorted, cmap="viridis", cbar_kws={"label": "Normalized Activation"})

    plt.title(f"Feature Flow Waterfall: Top {heatmap_k} Concept Features")
    plt.xlabel("Generation Steps (Noise -> Image)")
    plt.ylabel("Features (Sorted by Peak Time)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_3_waterfall_flow.png"), dpi=150)
    plt.close()


    print(f"[exp53] 输出目录: {out_dir}")
    print(f"[exp53] Top+ (scores 最大) feature ids: {top_pos_ids[:10].tolist()}")
