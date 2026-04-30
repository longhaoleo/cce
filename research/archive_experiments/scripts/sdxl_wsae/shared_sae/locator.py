"""SharedSAE 版概念定位正式入口。"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from sdxl_wsae.shared_sae.delta import DeltaExtractor
from sdxl_wsae.shared_sae.scoring import (
    _load_blacklist_ids,
    _load_concept_prompts_from_json,
    _saeuron_score_v2,
    _save_feature_time_scores_csv,
    _save_score_compare_csv,
    _save_topk_csv,
    _select_step_indices,
    _taris_score,
    _topk_with_blacklist,
)
from sdxl_wsae.utils import block_short_name, ensure_dir, safe_name

from .common import (
    add_checkpoint_args,
    add_generation_override_args,
    add_model_args,
    coords_for_hw,
    load_hooked_pipeline,
    load_shared_checkpoint_bundle,
    make_generator,
    resolve_blocks,
    resolve_checkpoint_dir,
    resolve_device_dtype,
    resolve_dtype,
    resolve_generation_hparams,
    resolve_norm_scale_by_block,
    scheduler_timesteps,
)

from SAE import SharedSAE


LOG_PREFIX = "shared-exp53"


@dataclass
class ActivationStats:
    """概念正负样本聚合后的激活统计。"""

    mean_by_block: Dict[str, torch.Tensor]
    std_by_block: Dict[str, torch.Tensor]
    count: int
    timesteps: List[int]


def build_parser() -> argparse.ArgumentParser:
    """构建 SharedSAE 概念定位参数。"""
    parser = argparse.ArgumentParser(description="SharedSAE version of exp53 concept locator")
    g_main = parser.add_argument_group("主参数")
    g_ckpt = parser.add_argument_group("SharedSAE checkpoint")
    g_model = parser.add_argument_group("SDXL")
    g_run = parser.add_argument_group("采样")
    g_score = parser.add_argument_group("概念评分")

    g_main.add_argument(
        "--concept_dir",
        type=str,
        default="./target_concept_dict",
        help="概念 json 文件夹（每个 json 文件名就是 concept_name）。",
    )
    g_main.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只跑这些 concept 名；为空表示扫描整个 concept_dir。",
    )
    g_main.add_argument(
        "--blocks",
        nargs="+",
        type=str,
        default=None,
        help="要定位的 block 列表；默认继承 checkpoint 配置。",
    )
    g_main.add_argument(
        "--max_prompts_per_side",
        type=int,
        default=0,
        help=">0 时仅取每个概念的前 N 条正样本和前 N 条负样本，适合 quick-check。",
    )
    g_main.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的概念与 block，不实际跑采样。",
    )

    add_checkpoint_args(g_ckpt)
    add_model_args(g_model)
    add_generation_override_args(g_run, prompt_required=False)

    g_score.add_argument("--taris_t_start", type=int, default=1000, help="时间窗口上界（高噪声侧）。")
    g_score.add_argument("--taris_t_end", type=int, default=0, help="时间窗口下界（低噪声侧）。")
    g_score.add_argument("--taris_num_steps", type=int, default=50, help="在窗口内均匀采样多少个 step。")
    g_score.add_argument("--taris_delta", type=float, default=1e-6, help="TARIS 分母平滑项。")
    g_score.add_argument("--taris_top_k", type=int, default=10, help="保存 top-k 概念特征。")
    g_score.add_argument(
        "--taris_score_mode",
        type=str,
        default="saeuron",
        choices=["taris", "saeuron"],
        help="主得分模式。",
    )
    g_score.add_argument("--taris_saeuron_eps", type=float, default=1e-6, help="SAeUron 分母平滑项。")
    g_score.add_argument(
        "--taris_compare_scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否同时导出 TARIS 与 SAeUron 的对比结果。",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    return build_parser().parse_args()


def _list_concepts(concept_dir: str) -> List[str]:
    """列出概念名称。"""
    root = Path(os.path.expanduser(str(concept_dir))).resolve()
    if not root.exists():
        raise FileNotFoundError(f"concept_dir 不存在: {root}")
    return [path.stem for path in sorted(root.glob("*.json")) if path.stem and not path.stem.startswith(".")]


def _truncate_prompts(prompts: Sequence[str], limit: int) -> List[str]:
    """按 quick-check 配额截断 prompt。"""
    items = list(prompts)
    if int(limit) <= 0:
        return items
    return items[: int(limit)]


def _resolve_blacklist_ids(*, block: str, out_dir: str) -> set[int]:
    """合并概念目录黑名单与全局高频特征黑名单。"""
    local_path = os.path.join(out_dir, "feature_blacklist.txt")
    global_path = os.path.join("concept_dict_freq", block_short_name(str(block)), "feature_blacklist.txt")
    ids = set(_load_blacklist_ids(local_path))
    ids.update(_load_blacklist_ids(global_path))
    return ids


@torch.no_grad()
def _prompt_activation_mats(
    *,
    pipe,
    model: SharedSAE,
    blocks: Sequence[str],
    norm_scale_by_block: Dict[str, float],
    prompt: str,
    steps: int,
    guidance_scale: float,
    resolution: int,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """对单个 prompt 跑一次轨迹，返回每个 block 的 [steps, n_features] 激活矩阵。"""
    _, cache = pipe.run_with_cache(
        prompt=str(prompt),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=make_generator(int(seed)),
        positions_to_cache=list(blocks),
        save_input=True,
        save_output=True,
        output_type="latent",
        height=int(resolution),
        width=int(resolution),
    )
    timesteps = scheduler_timesteps(pipe)
    extractor = DeltaExtractor()
    params = next(model.parameters())

    mats_by_block: Dict[str, torch.Tensor] = {}
    for block in blocks:
        rows: List[torch.Tensor] = []
        for item in extractor.extract(block=str(block), cache=cache, timesteps=timesteps):
            x = item.x.to(device=params.device, dtype=params.dtype)
            coords = coords_for_hw(
                hw=item.hw,
                n_tokens=int(x.shape[0]),
                device=params.device,
                dtype=params.dtype,
            )
            timestep_t = torch.tensor([float(item.timestep)], device=params.device, dtype=params.dtype)
            cache_out = model(
                x * float(norm_scale_by_block.get(str(block), 1.0)),
                block_name=str(block),
                timestep=timestep_t,
                coords_norm=coords,
                use_out_adapter=False,
                update_dead_stats=False,
            )
            rows.append(cache_out.z.mean(dim=0).detach().float().cpu())
        mats_by_block[str(block)] = torch.stack(rows, dim=0)
    return mats_by_block, timesteps


@torch.no_grad()
def _compute_activation_stats(
    *,
    pipe,
    model: SharedSAE,
    blocks: Sequence[str],
    norm_scale_by_block: Dict[str, float],
    prompts: Sequence[str],
    steps: int,
    guidance_scale: float,
    resolution: int,
    seed: int,
    track_std: bool,
) -> ActivationStats:
    """对一组 prompt 累积每个 block 的均值和标准差。"""
    prompt_list = [str(x) for x in prompts]
    if not prompt_list:
        raise ValueError("prompts 不能为空。")

    first_mats, timesteps = _prompt_activation_mats(
        pipe=pipe,
        model=model,
        blocks=blocks,
        norm_scale_by_block=norm_scale_by_block,
        prompt=prompt_list[0],
        steps=steps,
        guidance_scale=guidance_scale,
        resolution=resolution,
        seed=seed,
    )
    mean_by_block = {block: mat.clone() for block, mat in first_mats.items()}
    m2_by_block = {
        block: torch.zeros_like(mat) for block, mat in first_mats.items()
    } if bool(track_std) else {}
    count = 1

    for prompt in prompt_list[1:]:
        mats_by_block, cur_timesteps = _prompt_activation_mats(
            pipe=pipe,
            model=model,
            blocks=blocks,
            norm_scale_by_block=norm_scale_by_block,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            resolution=resolution,
            seed=seed,
        )
        if list(cur_timesteps) != list(timesteps):
            raise ValueError("不同 prompt 的 scheduler timesteps 不一致。")
        count += 1
        for block in blocks:
            mat = mats_by_block[str(block)]
            prev_mean = mean_by_block[str(block)]
            delta = mat - prev_mean
            new_mean = prev_mean + delta / float(count)
            mean_by_block[str(block)] = new_mean
            if bool(track_std):
                m2_by_block[str(block)] = m2_by_block[str(block)] + delta * (mat - new_mean)

    std_by_block: Dict[str, torch.Tensor] = {}
    for block in blocks:
        mean = mean_by_block[str(block)]
        if bool(track_std) and count > 1:
            var = m2_by_block[str(block)] / float(count)
            std_by_block[str(block)] = torch.sqrt(torch.clamp(var, min=0.0))
        else:
            std_by_block[str(block)] = torch.zeros_like(mean)
    return ActivationStats(
        mean_by_block=mean_by_block,
        std_by_block=std_by_block,
        count=count,
        timesteps=list(timesteps),
    )


def _save_meta(
    *,
    out_dir: str,
    block: str,
    concept_name_raw: str,
    concept_root: str,
    concept_json_path: Path,
    pos_prompts: Sequence[str],
    neg_prompts: Sequence[str],
    step_indices: Sequence[int],
    timesteps: Sequence[int],
    score_label: str,
    enable_compare: bool,
    saeuron_eps: float,
    delta: float,
    ckpt_dir: Path,
) -> None:
    """保存概念统计的伴随说明。"""
    meta_path = os.path.join(out_dir, "taris_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"shared_sae_checkpoint={ckpt_dir}\n")
        f.write(f"block={block}\n")
        f.write(f"concept_name={concept_name_raw}\n")
        f.write(f"concept_root={concept_root}\n")
        f.write(f"concept_json={concept_json_path}\n")
        f.write(f"pos_prompts={list(pos_prompts)}\n")
        f.write(f"neg_prompts={list(neg_prompts)}\n")
        f.write(f"score_mode={score_label}\n")
        f.write(f"enable_score_compare={enable_compare}\n")
        f.write(f"saeuron_eps={saeuron_eps}\n")
        f.write(f"delta={float(delta)}\n")
        f.write(f"selected_step_indices={list(map(int, step_indices))}\n")
        f.write(f"selected_timesteps={[int(timesteps[idx]) for idx in step_indices]}\n")
        f.write("feature_time_scores.diff=normalized_diff(pos_mu/sum_pos - neg_mu/sum_neg)\n")


def main() -> None:
    """程序主入口。"""
    args = parse_args()
    ckpt_dir = resolve_checkpoint_dir(ckpt_dir=str(args.ckpt_dir), output_root=str(args.output_root))
    print(f"[{LOG_PREFIX}] 使用 checkpoint: {ckpt_dir}")

    concept_names = _list_concepts(str(args.concept_dir))
    only = set(map(str, args.only)) if args.only else None
    if only is not None:
        concept_names = [name for name in concept_names if name in only]
    if not concept_names:
        raise ValueError("筛选后 concept 为空，请检查 --only 或 concept_dir。")

    device, dtype = resolve_device_dtype(str(args.device), resolve_dtype(str(args.dtype)), log_prefix=LOG_PREFIX)
    bundle = load_shared_checkpoint_bundle(ckpt_dir=ckpt_dir, device=device, dtype=dtype)
    blocks = resolve_blocks(requested_blocks=args.blocks, ckpt_cfg=bundle.config)
    norm_scale_by_block = resolve_norm_scale_by_block(
        bundle=bundle,
        blocks=blocks,
        log_prefix=LOG_PREFIX,
        warn_if_missing=False,
    )
    steps, guidance_scale, resolution = resolve_generation_hparams(args=args, ckpt_cfg=bundle.config)

    print(f"[{LOG_PREFIX}] concepts={len(concept_names)} blocks={len(blocks)}")
    for concept_name in concept_names:
        print(f"[{LOG_PREFIX}] 计划运行 concept={concept_name} blocks={blocks}")
    if bool(args.dry_run):
        return

    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix=LOG_PREFIX,
    )

    for concept_name_raw in concept_names:
        pos_prompts, neg_prompts, _raw_json, concept_json_path = _load_concept_prompts_from_json(
            concept_name=str(concept_name_raw),
            concept_root=str(args.concept_dir),
        )
        pos_prompts = _truncate_prompts(pos_prompts, int(args.max_prompts_per_side))
        neg_prompts = _truncate_prompts(neg_prompts, int(args.max_prompts_per_side))
        if not pos_prompts or not neg_prompts:
            raise ValueError(f"概念 {concept_name_raw} 截断后 pos/neg 为空。")

        print(f"[{LOG_PREFIX}] run concept={concept_name_raw} pos={len(pos_prompts)} neg={len(neg_prompts)}")
        pos_stats = _compute_activation_stats(
            pipe=pipe,
            model=bundle.model,
            blocks=blocks,
            norm_scale_by_block=norm_scale_by_block,
            prompts=pos_prompts,
            steps=steps,
            guidance_scale=guidance_scale,
            resolution=resolution,
            seed=int(args.seed),
            track_std=False,
        )
        neg_stats = _compute_activation_stats(
            pipe=pipe,
            model=bundle.model,
            blocks=blocks,
            norm_scale_by_block=norm_scale_by_block,
            prompts=neg_prompts,
            steps=steps,
            guidance_scale=guidance_scale,
            resolution=resolution,
            seed=int(args.seed),
            track_std=True,
        )
        if list(pos_stats.timesteps) != list(neg_stats.timesteps):
            raise ValueError("正负样本的 scheduler timesteps 不一致。")

        step_indices = _select_step_indices(
            pos_stats.timesteps,
            t_start=int(args.taris_t_start),
            t_end=int(args.taris_t_end),
            num_t_samples=int(args.taris_num_steps),
        )
        score_mode = str(args.taris_score_mode).strip().lower()
        enable_compare = bool(args.taris_compare_scores)
        saeuron_eps = float(args.taris_saeuron_eps)

        for block in blocks:
            pos_mu = pos_stats.mean_by_block[str(block)]
            neg_mu = neg_stats.mean_by_block[str(block)]
            neg_std = neg_stats.std_by_block[str(block)]

            need_taris = (score_mode == "taris") or enable_compare
            need_saeuron = (score_mode == "saeuron") or enable_compare
            taris_scores = (
                _taris_score(
                    pos_mu=pos_mu,
                    neg_mu=neg_mu,
                    step_indices=step_indices,
                    delta=float(args.taris_delta),
                )
                if need_taris
                else None
            )
            saeuron_scores = (
                _saeuron_score_v2(
                    pos_mu=pos_mu,
                    neg_mu=neg_mu,
                    neg_std=neg_std,
                    step_indices=step_indices,
                    epsilon=saeuron_eps,
                )
                if need_saeuron
                else None
            )

            if score_mode == "taris":
                assert taris_scores is not None
                scores_primary = taris_scores
                score_label = "taris"
            else:
                assert saeuron_scores is not None
                scores_primary = saeuron_scores
                score_label = "saeuron"

            out_dir = os.path.join("concept_dict", block_short_name(str(block)), safe_name(str(concept_name_raw)))
            ensure_dir(out_dir)

            blacklist_ids = _resolve_blacklist_ids(block=str(block), out_dir=out_dir)
            top_pos_ids, top_pos_vals, top_neg_ids, top_neg_vals = _topk_with_blacklist(
                scores=scores_primary,
                top_k=int(args.taris_top_k),
                blacklist_ids=blacklist_ids,
            )

            taris_top = None
            saeuron_top = None
            if taris_scores is not None:
                taris_top = _topk_with_blacklist(
                    scores=taris_scores,
                    top_k=int(args.taris_top_k),
                    blacklist_ids=blacklist_ids,
                )
            if saeuron_scores is not None:
                saeuron_top = _topk_with_blacklist(
                    scores=saeuron_scores,
                    top_k=int(args.taris_top_k),
                    blacklist_ids=blacklist_ids,
                )

            _save_meta(
                out_dir=out_dir,
                block=str(block),
                concept_name_raw=str(concept_name_raw),
                concept_root=str(args.concept_dir),
                concept_json_path=concept_json_path,
                pos_prompts=pos_prompts,
                neg_prompts=neg_prompts,
                step_indices=step_indices,
                timesteps=pos_stats.timesteps,
                score_label=score_label,
                enable_compare=enable_compare,
                saeuron_eps=saeuron_eps,
                delta=float(args.taris_delta),
                ckpt_dir=bundle.ckpt_dir,
            )
            _save_topk_csv(os.path.join(out_dir, "top_positive_features.csv"), top_ids=top_pos_ids, top_vals=top_pos_vals)
            _save_topk_csv(os.path.join(out_dir, "top_negative_features.csv"), top_ids=top_neg_ids, top_vals=top_neg_vals)

            if enable_compare and taris_top is not None and saeuron_top is not None:
                taris_pos_ids, taris_pos_vals, taris_neg_ids, taris_neg_vals = taris_top
                saeuron_pos_ids, saeuron_pos_vals, saeuron_neg_ids, saeuron_neg_vals = saeuron_top
                _save_topk_csv(os.path.join(out_dir, "top_positive_features_taris.csv"), top_ids=taris_pos_ids, top_vals=taris_pos_vals)
                _save_topk_csv(os.path.join(out_dir, "top_negative_features_taris.csv"), top_ids=taris_neg_ids, top_vals=taris_neg_vals)
                _save_topk_csv(os.path.join(out_dir, "top_positive_features_saeuron.csv"), top_ids=saeuron_pos_ids, top_vals=saeuron_pos_vals)
                _save_topk_csv(os.path.join(out_dir, "top_negative_features_saeuron.csv"), top_ids=saeuron_neg_ids, top_vals=saeuron_neg_vals)
                _save_score_compare_csv(
                    os.path.join(out_dir, "score_compare_taris_vs_saeuron.csv"),
                    taris_scores=taris_scores,
                    saeuron_scores=saeuron_scores,
                )

            _save_feature_time_scores_csv(
                os.path.join(out_dir, "feature_time_scores.csv"),
                timesteps=pos_stats.timesteps,
                feature_ids=top_pos_ids.tolist(),
                scores_primary=scores_primary,
                score_mode=score_label,
                pos_mu=pos_mu,
                neg_mu=neg_mu,
                neg_std=neg_std,
                taris_scores=taris_scores,
                saeuron_scores=saeuron_scores,
                saeuron_eps=saeuron_eps,
                delta=float(args.taris_delta),
            )

            torch.save(
                {
                    "shared_sae_checkpoint": str(bundle.ckpt_dir),
                    "block": str(block),
                    "concept_name": str(concept_name_raw),
                    "pos_prompts": list(pos_prompts),
                    "neg_prompts": list(neg_prompts),
                    "timesteps": list(pos_stats.timesteps),
                    "selected_step_indices": list(map(int, step_indices)),
                    "score_mode": score_label,
                    "enable_score_compare": enable_compare,
                    "saeuron_eps": saeuron_eps,
                    "scores_primary": scores_primary,
                    "scores_taris": taris_scores,
                    "scores_saeuron": saeuron_scores,
                    "top_positive_ids": top_pos_ids,
                    "top_positive_vals": top_pos_vals,
                    "top_negative_ids": top_neg_ids,
                    "top_negative_vals": top_neg_vals,
                    "pos_mu": pos_mu,
                    "neg_mu": neg_mu,
                    "neg_std": neg_std,
                },
                os.path.join(out_dir, "taris_dump.pt"),
            )
            print(
                f"[{LOG_PREFIX}] saved concept={concept_name_raw} block={block} "
                f"top1={int(top_pos_ids[0].item()) if int(top_pos_ids.numel()) > 0 else -1}"
            )


if __name__ == "__main__":
    main()
