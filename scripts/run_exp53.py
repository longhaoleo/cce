#!/usr/bin/env python3
"""
批量运行 exp53（TARIS）：

- 概念 prompts 从 `target_concept_dict/{concept_name}.json` 读取
- 该脚本只负责“枚举概念并逐个调用 exp53”，不再维护内置 CONCEPTS 表
- 输出目录由 exp53 固定到 `out_concept_dict/{concept_name}/`

用法示例：
  python scripts/run_exp53.py
  python scripts/run_exp53.py --only red glasses
  python scripts/run_exp53.py --concept_dir ./target_concept_dict --loc_block unet.mid_block.attentions.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """保证从任意工作目录运行时，都能导入仓库内模块（尤其是顶层的 `SAE/`）。"""
    repo_root = Path(__file__).resolve().parents[1]
    scripts_root = repo_root / "scripts"
    for p in (str(repo_root), str(scripts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_path()

from sdxl_wsae.configs import ConceptLocateConfig, ModelConfig, RunConfig, SAEConfig  # noqa: E402
from sdxl_wsae.core.session import SDXLExperimentSession  # noqa: E402
from sdxl_wsae.experiments.exp53_concept_locator_taris import run_exp53_concept_locator_taris  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch run exp53 (TARIS) for concepts in a json folder")
    ap.add_argument(
        "--concept_dir",
        type=str,
        default="./target_concept_dict",
        help="概念 json 文件夹（每个 json 文件名就是 concept_name）",
    )

    ap.add_argument("--sae_root", type=str, default="~/autodl-tmp/sdxl-saes")
    ap.add_argument("--model_id", type=str, default="~/autodl-tmp/models/sd-xl-base-1.0-fp16-only")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--prefer_k", type=int, default=10)
    ap.add_argument("--prefer_hidden", type=int, default=5120)

    ap.add_argument("--loc_block", type=str, default="unet.mid_block.attentions.0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--taris_t_start", type=int, default=800)
    ap.add_argument("--taris_t_end", type=int, default=200)
    ap.add_argument("--taris_num_steps", type=int, default=10)
    ap.add_argument("--taris_delta", type=float, default=1e-6)
    ap.add_argument("--taris_top_k", type=int, default=50)

    ap.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="只跑这些概念名（空表示全跑），例如 --only red glasses",
    )
    return ap.parse_args()


def _list_concepts(concept_dir: str) -> list[str]:
    """列出 concept_name 列表（来自 `concept_dir/*.json`）。"""
    root = Path(os.path.expanduser(concept_dir)).resolve()
    if not root.exists():
        raise FileNotFoundError(f"concept_dir 不存在: {root}")
    names = [p.stem for p in sorted(root.glob("*.json"))]
    return [n for n in names if n and not n.startswith(".")]


def main() -> None:
    args = parse_args()

    concept_names = _list_concepts(args.concept_dir)
    if not concept_names:
        raise FileNotFoundError(f"concept_dir 下没有任何 json: {args.concept_dir}")

    only = set(map(str, args.only)) if args.only else None
    if only is not None:
        concept_names = [n for n in concept_names if n in only]
    if not concept_names:
        raise ValueError("筛选后 concept 为空（检查 --only 或 concept_dir 下的文件名）。")

    model_cfg = ModelConfig(
        model_id=args.model_id,
        device=args.device,
        dtype_name=args.dtype,
    )
    sae_cfg = SAEConfig(
        sae_root=args.sae_root,
        blocks=(str(args.loc_block),),  # exp53 只需要一个 block
        prefer_k=int(args.prefer_k),
        prefer_hidden=int(args.prefer_hidden),
    )
    # run_cfg.prompt 会在 exp53 内部被替换成每条 prompt，这里给个占位即可
    run_cfg = RunConfig(
        prompt="",
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
    )

    # 复用同一个 session：避免每个概念重复加载模型和 SAE
    session = SDXLExperimentSession(model_cfg, sae_cfg)

    for concept_name in concept_names:
        # 注意：exp53 会按 concept_name 去 `target_concept_dict/{concept_name}.json` 读取 prompts
        # 如果你这里 concept_dir 不是默认值，请确保 concept_dir 与 exp53 使用的文件夹一致。
        concept_cfg = ConceptLocateConfig(
            block=str(args.loc_block),
            concept_name=str(concept_name),
            t_start=int(args.taris_t_start),
            t_end=int(args.taris_t_end),
            num_t_samples=int(args.taris_num_steps),
            delta=float(args.taris_delta),
            top_k=int(args.taris_top_k),
        )

        print(f"[exp53] run concept={concept_name}")
        run_exp53_concept_locator_taris(
            model_cfg,
            sae_cfg,
            run_cfg,
            concept_cfg,
            output_dir=".",  # exp53 当前固定输出到 out_concept_dict/{concept_name}/
            session=session,
        )


if __name__ == "__main__":
    main()

