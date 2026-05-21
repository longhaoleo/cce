"""SharedSAE 批量概念擦除主逻辑。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .features.intervention import _build_block_scale_map, _save_hook_debug_csv
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
from .erase import (
    add_intervention_args,
    SharedInterventionConfig,
    _build_intervention_spec,
    _resolve_injectconcept,
    _resolve_feature_bundle,
    _scale_coeff_by_step,
    _save_eval_pair,
    _save_time_weight_debug_csv,
    build_intervention_cfg_from_args,
    build_shared_feature_intervention_hook,
    resolve_intervention_roots,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="Batch SharedSAE concept erasure runner")
    g_io = ap.add_argument_group("输入/输出")
    g_ckpt = ap.add_argument_group("Shared checkpoint")
    g_model = ap.add_argument_group("SDXL")
    g_run = ap.add_argument_group("采样")
    g_int = ap.add_argument_group("干预")

    g_io.add_argument("--prompts_path", type=str, default="./batch_test_prompt/car.csv", help="prompt 集合路径（.txt / .csv）")
    g_io.add_argument("--output_dir", type=str, default="./image_output/batch_shared_concept_erase")
    g_io.add_argument("--concepts", nargs="+", type=str, default=None, help="概念列表；不传则自动扫描 target_concept_dict/*.json")
    g_io.add_argument("--target_concept_dict_dir", type=str, default="./target_concept_dict")
    g_io.add_argument("--concept_root", type=str, default="concept_dict", help="Shared 概念统计根目录。")
    g_io.add_argument("--sae_root", type=str, default="", help="统一 SAE 产物根目录；传入后自动映射 concept-dig / blacklist 等子目录。")
    g_io.add_argument("--from_case", type=int, default=0)
    g_io.add_argument("--till_case", type=int, default=1_000_000)
    g_io.add_argument("--max_prompts", type=int, default=0, help=">0 时只取前 N 条")
    g_io.add_argument("--dry_run", action="store_true", help="只解析并打印计划，不执行生成")
    g_io.add_argument("--base_seed", type=int, default=42, help="当 prompt 行没有 seed 时使用 base_seed+idx")

    add_checkpoint_args(g_ckpt)
    add_model_args(g_model)
    add_generation_override_args(g_run, prompt_required=False)
    g_run.add_argument("--no_baseline", action="store_true", help="只生成干预图，不生成 baseline。")

    add_intervention_args(g_int, include_targetconcept=False)
    return ap.parse_args()


def _discover_concepts(target_dir: str) -> List[str]:
    """自动扫描概念名。"""
    p = Path(os.path.expanduser(target_dir))
    if not p.exists():
        raise FileNotFoundError(f"concept dict 目录不存在: {p}")
    names = sorted(x.stem for x in p.glob("*.json"))
    if not names:
        raise ValueError(f"concept dict 目录下没有 json: {p}")
    return names


def _load_prompts(path: str, *, base_seed: int) -> List[Dict[str, object]]:
    """读取 prompt 列表。"""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        raise FileNotFoundError(f"prompts_path 不存在: {p}")

    rows: List[Dict[str, object]] = []
    if p.suffix.lower() == ".txt":
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                prompt = line.strip()
                if not prompt:
                    continue
                rows.append({"case_number": idx, "prompt": prompt, "seed": int(base_seed + idx)})
        return rows

    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            fieldnames = [str(x).strip() for x in header]
            if not fieldnames:
                raise ValueError("CSV 为空或没有表头。")

            prompt_col = "prompt" if "prompt" in fieldnames else fieldnames[0]
            has_case = "case_number" in fieldnames
            has_seed = "evaluation_seed" in fieldnames or "seed" in fieldnames
            seed_col = "evaluation_seed" if "evaluation_seed" in fieldnames else "seed"
            prompt_idx = fieldnames.index(prompt_col)
            case_idx = fieldnames.index("case_number") if has_case else -1
            seed_idx = fieldnames.index(seed_col) if has_seed else -1

            for idx, row in enumerate(reader):
                if not row:
                    continue
                cells = [str(x) for x in row]
                prompt = ""
                case_number = idx
                seed = int(base_seed + idx)

                # 兼容 prompt 内含未转义逗号的常见格式：
                # case_number,prompt,evaluation_seed
                # 这时首列为 case，末列为 seed，中间所有列都属于 prompt。
                if has_case and has_seed and prompt_idx == 1 and case_idx == 0 and seed_idx == 2 and len(cells) >= 3:
                    case_raw = cells[0].strip()
                    seed_raw = cells[-1].strip()
                    prompt = ",".join(cells[1:-1]).strip()
                    if case_raw:
                        case_number = int(case_raw)
                    if seed_raw:
                        seed = int(seed_raw)
                else:
                    if prompt_idx >= len(cells):
                        continue
                    prompt = str(cells[prompt_idx]).strip()
                    if has_case and case_idx >= 0 and case_idx < len(cells) and str(cells[case_idx]).strip():
                        case_number = int(cells[case_idx])
                    if has_seed and seed_idx >= 0 and seed_idx < len(cells) and str(cells[seed_idx]).strip():
                        seed = int(cells[seed_idx])

                if not prompt:
                    continue
                rows.append({"case_number": case_number, "prompt": prompt, "seed": seed})
        return rows

    raise ValueError(f"仅支持 .txt / .csv，当前为: {p.suffix}")


def _build_intervention_cfg(args: argparse.Namespace) -> SharedInterventionConfig:
    """从 CLI 参数构造 Shared 干预配置。"""
    return build_intervention_cfg_from_args(args)


def _save_compare_image(*, output_dir: str, baseline_img, steered_img) -> None:
    """保存 baseline / steered 横向对比图。"""
    if baseline_img is None or steered_img is None:
        return
    from PIL import Image

    compare = Image.new("RGB", (baseline_img.width + steered_img.width, max(baseline_img.height, steered_img.height)))
    compare.paste(baseline_img, (0, 0))
    compare.paste(steered_img, (baseline_img.width, 0))
    compare_path = Path(output_dir).expanduser().resolve() / "intervention_compare.png"
    compare.save(compare_path)


def main() -> None:
    """程序主入口。"""
    args = parse_args()

    all_prompts = _load_prompts(args.prompts_path, base_seed=int(args.base_seed))
    prompts = [
        r for r in all_prompts if int(r["case_number"]) >= int(args.from_case) and int(r["case_number"]) <= int(args.till_case)
    ]
    if int(args.max_prompts) > 0:
        prompts = prompts[: int(args.max_prompts)]
    if not prompts:
        raise ValueError("过滤后 prompts 为空，请检查 from/till/max_prompts。")

    concepts = [str(x) for x in args.concepts] if args.concepts else _discover_concepts(args.target_concept_dict_dir)
    if not concepts:
        raise ValueError("concepts 为空。")

    out_root = os.path.expanduser(str(args.output_dir))
    ensure_dir(out_root)

    if bool(args.dry_run):
        print(f"[batch-shared] dry_run prompts={len(prompts)} concepts={len(concepts)} total_jobs={len(prompts) * len(concepts)}")
        return

    ckpt_dir = resolve_checkpoint_dir(ckpt_dir=str(args.ckpt_dir), output_root=str(args.output_root))
    device, dtype = resolve_device_dtype(str(args.device), resolve_dtype(str(args.dtype)), log_prefix="batch-shared")
    bundle = load_shared_checkpoint_bundle(ckpt_dir=ckpt_dir, device=device, dtype=dtype)
    blocks = resolve_blocks(requested_blocks=args.blocks, ckpt_cfg=bundle.config)
    norm_scale_by_block = resolve_norm_scale_by_block(
        bundle=bundle,
        blocks=blocks,
        log_prefix="batch-shared",
        warn_if_missing=True,
    )
    steps, guidance_scale, resolution = resolve_generation_hparams(args=args, ckpt_cfg=bundle.config)
    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix="batch-shared",
    )
    intervention_cfg = _build_intervention_cfg(args)
    mode = str(intervention_cfg.mode).lower()
    injectconcept = _resolve_injectconcept(args, mode=mode)
    concept_root, concept_dict_freq_root = resolve_intervention_roots(args)

    manifest_path = os.path.join(out_root, "run_manifest.jsonl")
    print(f"[batch-shared] checkpoint={ckpt_dir}")
    print(f"[batch-shared] prompts={len(prompts)} concepts={len(concepts)} total_jobs={len(prompts) * len(concepts)}")
    print(f"[batch-shared] blocks={blocks}")
    print(f"[batch-shared] output_dir={out_root}")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        total = len(prompts) * len(concepts)
        done = 0
        for concept in concepts:
            concept_dir = os.path.join(out_root, safe_name(concept))
            ensure_dir(concept_dir)

            features_by_block: Dict[str, object] = {}
            inject_features_by_block: Dict[str, object] = {}
            for block in blocks:
                features_by_block[str(block)] = _resolve_feature_bundle(
                    block=str(block),
                    targetconcept=str(concept),
                    concept_root=str(concept_root),
                    top_k=int(intervention_cfg.feature_top_k),
                    total_steps=int(steps),
                    use_time_weight=bool(intervention_cfg.time.use_stat_weight),
                    blacklist_root=str(concept_dict_freq_root),
                )
                if mode == "replace":
                    inject_features_by_block[str(block)] = _resolve_feature_bundle(
                        block=str(block),
                        targetconcept=str(injectconcept),
                        concept_root=str(concept_root),
                        top_k=int(intervention_cfg.feature_top_k),
                        total_steps=int(steps),
                        use_time_weight=bool(intervention_cfg.time.use_stat_weight),
                        blacklist_root=str(concept_dict_freq_root),
                    )
            coeffs_by_block = {
                block: _scale_coeff_by_step(
                    feat.coeff_by_step,
                    scale=float(intervention_cfg.time.stat_weight_scale) if bool(intervention_cfg.time.use_stat_weight) else 1.0,
                )
                for block, feat in features_by_block.items()
            }
            inject_coeffs_by_block = {
                block: _scale_coeff_by_step(
                    feat.coeff_by_step,
                    scale=float(intervention_cfg.time.stat_weight_scale) if bool(intervention_cfg.time.use_stat_weight) else 1.0,
                )
                for block, feat in inject_features_by_block.items()
            }
            block_scale_map = _build_block_scale_map(
                blocks=[str(block) for block in blocks],
                base_scale=float(intervention_cfg.scale),
                coeffs_by_block=coeffs_by_block,
            )
            inject_block_scale_map = None
            if mode == "replace":
                inject_block_scale_map = _build_block_scale_map(
                    blocks=[str(block) for block in blocks],
                    base_scale=float(intervention_cfg.inject_scale),
                    coeffs_by_block=inject_coeffs_by_block,
                )

            for row in prompts:
                done += 1
                case_number = int(row["case_number"])
                prompt = str(row["prompt"])
                seed = int(row["seed"])
                prompt_dir = os.path.join(concept_dir, f"case_{case_number:06d}")
                ensure_dir(prompt_dir)
                print(f"[{done}/{total}] concept={concept} case={case_number} seed={seed}")

                hooks = {}
                for block in blocks:
                    spec = _build_intervention_spec(
                        block=str(block),
                        features=features_by_block[str(block)],
                        inject_features=inject_features_by_block.get(str(block)),
                        cfg=intervention_cfg,
                        total_steps=int(steps),
                        block_scale_map=block_scale_map,
                        inject_block_scale_map=inject_block_scale_map,
                    )
                    hooks[str(block)] = build_shared_feature_intervention_hook(
                        pipe=pipe,
                        sae=bundle.model,
                        spec=spec,
                        block_name=str(block),
                        block_norm_scale=float(norm_scale_by_block[str(block)]),
                        use_out_adapter_for_decode=bool(intervention_cfg.use_out_adapter_for_decode),
                    )

                status = "ok"
                err = ""
                baseline_img = None
                steered_img = None
                try:
                    if not bool(args.no_baseline):
                        baseline_out = pipe(
                            prompt=prompt,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance_scale),
                            generator=make_generator(seed),
                            output_type="pil",
                            height=int(resolution),
                            width=int(resolution),
                        )
                        baseline_img = extract_first_image(baseline_out)
                        if baseline_img is not None:
                            baseline_img.save(Path(prompt_dir) / "intervention_baseline.png")

                    steered_out = pipe.run_with_hooks(
                        prompt=prompt,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        generator=make_generator(seed),
                        output_type="pil",
                        height=int(resolution),
                        width=int(resolution),
                        position_hook_dict=hooks,
                    )
                    steered_img = extract_first_image(steered_out)
                    if steered_img is not None:
                        steered_img.save(Path(prompt_dir) / "intervention_steered.png")

                    _save_compare_image(output_dir=prompt_dir, baseline_img=baseline_img, steered_img=steered_img)
                    _save_hook_debug_csv(hooks=hooks, out_dir=str(prompt_dir), tag="shared_intervention")
                    _save_time_weight_debug_csv(hooks=hooks, out_dir=str(prompt_dir))
                    _save_eval_pair(
                        output_dir=str(prompt_dir),
                        case_number=case_number,
                        sample_name=str(concept),
                        baseline_img=baseline_img,
                        steered_img=steered_img,
                    )

                    per_case_manifest = {
                        "ckpt_dir": str(bundle.ckpt_dir),
                        "sae_root": str(getattr(args, "sae_root", "") or ""),
                        "concept_root": str(concept_root),
                        "concept_dict_freq_root": str(concept_dict_freq_root),
                        "prompt": prompt,
                        "targetconcept": str(concept),
                        "injectconcept": str(injectconcept),
                        "case_number": int(case_number),
                        "blocks": [str(block) for block in blocks],
                        "steps": int(steps),
                        "guidance_scale": float(guidance_scale),
                        "resolution": int(resolution),
                        "seed": int(seed),
                        "no_baseline": bool(args.no_baseline),
                        "intervention": asdict(intervention_cfg),
                        "norm_scale_by_block": norm_scale_by_block,
                        "block_scale_map": block_scale_map,
                        "inject_block_scale_map": {} if inject_block_scale_map is None else inject_block_scale_map,
                        "features_by_block": {block: feat.to_manifest() for block, feat in features_by_block.items()},
                        "inject_features_by_block": {
                            block: feat.to_manifest() for block, feat in inject_features_by_block.items()
                        },
                        "eval_original_dir": str((Path(prompt_dir).expanduser().resolve() / "eval_original")),
                        "eval_erased_dir": str((Path(prompt_dir).expanduser().resolve() / "eval_erased")),
                    }
                    with (Path(prompt_dir) / "run_manifest.json").open("w", encoding="utf-8") as f:
                        json.dump(per_case_manifest, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    status = "failed"
                    err = str(e)
                    print(f"[warn] failed: concept={concept} case={case_number} err={e}")

                rec = {
                    "status": status,
                    "concept": concept,
                    "case_number": case_number,
                    "seed": seed,
                    "prompt": prompt,
                    "output_dir": prompt_dir,
                    "error": err,
                    "sae_root": str(getattr(args, "sae_root", "") or ""),
                    "blocks": list(blocks),
                    "injectconcept": str(injectconcept),
                    "steps": int(steps),
                    "guidance_scale": float(guidance_scale),
                    "resolution": int(resolution),
                    "int_mode": str(intervention_cfg.mode),
                    "int_scale": float(intervention_cfg.scale),
                    "int_inject_scale": float(intervention_cfg.inject_scale),
                    "int_feature_top_k": int(intervention_cfg.feature_top_k),
                    "int_use_time_weight": bool(intervention_cfg.time.use_stat_weight),
                    "int_use_stat_time_weight": bool(intervention_cfg.time.use_stat_weight),
                    "int_use_learned_time_weight": bool(intervention_cfg.time.use_learned_weight),
                    "int_learned_time_weight_mode": str(intervention_cfg.time.learned_weight_mode),
                    "int_learned_time_weight_target_mean": float(intervention_cfg.time.learned_weight_target_mean),
                    "int_max_delta_over_x": float(intervention_cfg.max_delta_over_x),
                    "int_time_fuse_mode": str(intervention_cfg.time.fuse_mode),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()


if __name__ == "__main__":
    main()
