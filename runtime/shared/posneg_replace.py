"""用同一概念的正/负特征做 replace：减 pos，加 neg。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

from .erase import (
    LOG_PREFIX,
    FeatureBundle,
    _build_intervention_spec,
    _load_topk_feature_ids,
    _resolve_concept_dir,
    _save_eval_pair,
    _scale_coeff_by_step,
    build_intervention_cfg_from_args,
    build_shared_feature_intervention_hook,
    resolve_intervention_roots,
)
from .features.intervention import _build_block_scale_map, _interpolate_coeff_by_step, _load_coeff_by_step_from_exp53_csv, _save_hook_debug_csv
from .features.scoring import _load_blacklist_ids
from .io_utils import ensure_dir, extract_first_image, safe_name
from .pipeline import (
    add_checkpoint_args,
    add_generation_override_args,
    add_model_args,
    load_hooked_pipeline,
    load_shared_checkpoint_bundle,
    make_generator,
    resolve_blocks,
    resolve_checkpoint_dir,
    resolve_device_dtype,
    resolve_dtype,
    resolve_generation_hparams,
    resolve_norm_scale_by_block,
)


DEFAULT_BLOCKS = [
    "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    "unet.up_blocks.0.attentions.0",
    "unet.up_blocks.0.attentions.1",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-concept posneg replace: ablate pos features and inject neg features")
    g_io = parser.add_argument_group("输入输出")
    g_ckpt = parser.add_argument_group("SharedSAE checkpoint")
    g_model = parser.add_argument_group("SDXL")
    g_run = parser.add_argument_group("采样")
    g_int = parser.add_argument_group("干预")

    g_io.add_argument("--output_dir", type=str, default="image_output/shared_posneg_replace")
    g_io.add_argument("--case_number", type=int, default=0)
    g_io.add_argument("--sample_name", type=str, default="")
    g_io.add_argument("--concept", type=str, required=True, help="概念名，对应 concept-dig/<block>/<concept>/")
    g_io.add_argument("--concept_root", type=str, default="concept_dict", help="概念定位结果根目录。")
    g_io.add_argument("--sae_root", type=str, default="", help="统一 SAE 产物根目录。")
    g_io.add_argument("--source_rank_csv", type=str, default="top_positive_features.csv", help="用于去除的特征排序表。")
    g_io.add_argument("--inject_rank_csv", type=str, default="top_negative_features.csv", help="用于注入的特征排序表。")

    add_checkpoint_args(g_ckpt)
    add_model_args(g_model)
    add_generation_override_args(g_run, prompt_required=True)
    g_run.add_argument("--no_baseline", action="store_true")

    g_int.add_argument("--blocks", nargs="+", type=str, default=list(DEFAULT_BLOCKS))
    g_int.add_argument("--int_scale", type=float, default=5000.0, help="pos feature 去除强度。")
    g_int.add_argument("--int_inject_scale", type=float, default=100.0, help="neg feature 注入强度。")
    g_int.add_argument("--int_feature_top_k", type=int, default=10)
    g_int.add_argument("--int_projection_ridge", type=float, default=1e-4)
    g_int.add_argument("--int_use_time_weight", action=argparse.BooleanOptionalAction, default=True)
    g_int.add_argument("--concept_dict_freq_root", type=str, default="concept_dict_freq")
    g_int.add_argument("--int_t_start", type=int, default=1000)
    g_int.add_argument("--int_t_end", type=int, default=0)
    g_int.add_argument("--int_step_start", type=int, default=-1)
    g_int.add_argument("--int_step_end", type=int, default=-1)
    g_int.add_argument("--int_use_spatial_weight", action=argparse.BooleanOptionalAction, default=True)
    g_int.add_argument("--use_out_adapter_for_decode", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _resolve_feature_bundle_from_rank_csv(
    *,
    block: str,
    concept: str,
    concept_root: str,
    rank_csv_name: str,
    top_k: int,
    total_steps: int,
    use_time_weight: bool,
    blacklist_root: str,
) -> FeatureBundle:
    concept_dir = _resolve_concept_dir(block=block, targetconcept=concept, concept_root=concept_root)
    rank_csv = concept_dir / str(rank_csv_name)
    blacklist_ids = set(_load_blacklist_ids(str(concept_dir / "feature_blacklist.txt")))
    global_root = Path(str(blacklist_root)).expanduser()
    if not global_root.is_absolute():
        global_root = Path.cwd() / global_root
    global_blacklist = global_root / safe_name(block).replace("_", ".")  # fallback, overwritten below if needed
    global_blacklist = global_root / block.split("unet.")[-1].replace("_blocks.", "").replace(".attentions.", ".")
    # use canonical block short name by reusing existing directory resolution convention
    from .io_utils import block_short_name
    global_blacklist = global_root / block_short_name(block) / "feature_blacklist.txt"
    blacklist_ids.update(_load_blacklist_ids(str(global_blacklist)))

    raw_feature_ids = _load_topk_feature_ids(rank_csv, top_k)
    feature_ids = _load_topk_feature_ids(rank_csv, top_k, blacklist_ids=blacklist_ids)
    if blacklist_ids:
        print(f"[{LOG_PREFIX}] block={block} {rank_csv_name} 黑名单过滤后取 topK: {len(raw_feature_ids)} -> {len(feature_ids)}")
    if not feature_ids:
        raise ValueError(f"block={block} {rank_csv_name} 过滤后 feature_ids 为空。")

    coeff_by_step: Dict[int, torch.Tensor] = {}
    if bool(use_time_weight):
        coeff_csv = concept_dir / "feature_time_scores.csv"
        coeff_raw = _load_coeff_by_step_from_exp53_csv(csv_path=str(coeff_csv), feature_ids=feature_ids)
        if coeff_raw:
            coeff_by_step = _interpolate_coeff_by_step(coeff_by_step=coeff_raw, total_steps=int(total_steps))
        else:
            # top_negative_features 通常不在 feature_time_scores.csv 里。
            # 这里降级成“每个 step 权重=1”，避免 replace 直接报错。
            ones = torch.ones(len(feature_ids), dtype=torch.float32)
            coeff_by_step = {step: ones.clone() for step in range(int(total_steps))}
            print(
                f"[{LOG_PREFIX}] block={block} {rank_csv_name} 在 feature_time_scores.csv 中没有 step 权重，"
                "降级为全 step 常数注入。"
            )

    return FeatureBundle(
        feature_ids=[int(x) for x in feature_ids],
        feature_scales=[1.0 for _ in feature_ids],
        coeff_by_step=coeff_by_step,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.int_mode = "replace"
    cfg = build_intervention_cfg_from_args(args)
    concept_root, concept_dict_freq_root = resolve_intervention_roots(args)

    ckpt_dir = resolve_checkpoint_dir(ckpt_dir=str(args.ckpt_dir), output_root=str(args.output_root))
    device, dtype = resolve_device_dtype(str(args.device), resolve_dtype(str(args.dtype)), log_prefix=LOG_PREFIX)
    bundle = load_shared_checkpoint_bundle(ckpt_dir=ckpt_dir, device=device, dtype=dtype)
    blocks = resolve_blocks(requested_blocks=args.blocks, ckpt_cfg=bundle.config)
    norm_scale_by_block = resolve_norm_scale_by_block(bundle=bundle, blocks=blocks, log_prefix=LOG_PREFIX, warn_if_missing=True)
    steps, guidance_scale, resolution = resolve_generation_hparams(args=args, ckpt_cfg=bundle.config)

    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix=LOG_PREFIX,
    )

    pos_by_block: Dict[str, FeatureBundle] = {}
    neg_by_block: Dict[str, FeatureBundle] = {}
    for block in blocks:
        pos_by_block[str(block)] = _resolve_feature_bundle_from_rank_csv(
            block=str(block),
            concept=str(args.concept),
            concept_root=str(concept_root),
            rank_csv_name=str(args.source_rank_csv),
            top_k=int(cfg.feature_top_k),
            total_steps=int(steps),
            use_time_weight=bool(cfg.time.use_weight),
            blacklist_root=str(concept_dict_freq_root),
        )
        neg_by_block[str(block)] = _resolve_feature_bundle_from_rank_csv(
            block=str(block),
            concept=str(args.concept),
            concept_root=str(concept_root),
            rank_csv_name=str(args.inject_rank_csv),
            top_k=int(cfg.feature_top_k),
            total_steps=int(steps),
            use_time_weight=bool(cfg.time.use_weight),
            blacklist_root=str(concept_dict_freq_root),
        )

    pos_coeffs = {
        block: _scale_coeff_by_step(feat.coeff_by_step, scale=float(cfg.time.weight_scale) if bool(cfg.time.use_weight) else 1.0)
        for block, feat in pos_by_block.items()
    }
    neg_coeffs = {
        block: _scale_coeff_by_step(feat.coeff_by_step, scale=float(cfg.time.weight_scale) if bool(cfg.time.use_weight) else 1.0)
        for block, feat in neg_by_block.items()
    }
    pos_scale_map = _build_block_scale_map(blocks=[str(block) for block in blocks], base_scale=float(cfg.scale), coeffs_by_block=pos_coeffs)
    neg_scale_map = _build_block_scale_map(blocks=[str(block) for block in blocks], base_scale=float(cfg.inject_scale), coeffs_by_block=neg_coeffs)

    hooks = {}
    for block in blocks:
        spec = _build_intervention_spec(
            block=str(block),
            features=pos_by_block[str(block)],
            inject_features=neg_by_block[str(block)],
            cfg=cfg,
            total_steps=int(steps),
            block_scale_map=pos_scale_map,
            inject_block_scale_map=neg_scale_map,
        )
        hooks[str(block)] = build_shared_feature_intervention_hook(
            pipe=pipe,
            sae=bundle.model,
            spec=spec,
            block_name=str(block),
            block_norm_scale=float(norm_scale_by_block[str(block)]),
            use_out_adapter_for_decode=bool(cfg.use_out_adapter_for_decode),
        )

    ensure_dir(str(args.output_dir))
    baseline_img = None
    if not bool(args.no_baseline):
        baseline_out = pipe(
            prompt=str(args.prompt),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=make_generator(int(args.seed)),
            output_type="pil",
            height=int(resolution),
            width=int(resolution),
        )
        baseline_img = extract_first_image(baseline_out)
        if baseline_img is not None:
            baseline_img.save(Path(args.output_dir).expanduser().resolve() / "intervention_baseline.png")

    steered_out = pipe.run_with_hooks(
        prompt=str(args.prompt),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=make_generator(int(args.seed)),
        output_type="pil",
        height=int(resolution),
        width=int(resolution),
        position_hook_dict=hooks,
    )
    steered_img = extract_first_image(steered_out)
    if steered_img is not None:
        steered_img.save(Path(args.output_dir).expanduser().resolve() / "intervention_steered.png")

    _save_hook_debug_csv(hooks=hooks, out_dir=str(args.output_dir), tag="shared_posneg_replace")
    _save_eval_pair(
        output_dir=str(args.output_dir),
        case_number=int(args.case_number),
        sample_name=str(args.sample_name or args.concept),
        baseline_img=baseline_img,
        steered_img=steered_img,
    )

    manifest = {
        "ckpt_dir": str(bundle.ckpt_dir),
        "sae_root": str(getattr(args, "sae_root", "") or ""),
        "concept_root": str(concept_root),
        "concept_dict_freq_root": str(concept_dict_freq_root),
        "concept": str(args.concept),
        "source_rank_csv": str(args.source_rank_csv),
        "inject_rank_csv": str(args.inject_rank_csv),
        "prompt": str(args.prompt),
        "blocks": [str(block) for block in blocks],
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "resolution": int(resolution),
        "seed": int(args.seed),
        "intervention": asdict(cfg),
        "pos_scale_map": pos_scale_map,
        "neg_scale_map": neg_scale_map,
        "pos_features_by_block": {block: feat.to_manifest() for block, feat in pos_by_block.items()},
        "neg_features_by_block": {block: feat.to_manifest() for block, feat in neg_by_block.items()},
    }
    with (Path(args.output_dir).expanduser().resolve() / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
