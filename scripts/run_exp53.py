#!/usr/bin/env python3
"""
最简单的批量跑 exp53（TARIS）入口：概念直接写在一个 dict 里。

你只需要改：
- CONCEPTS：每个概念一组正负 prompts
- 下面 main() 里的默认参数（或用命令行覆盖）
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



# 概念表：key 是 concept_name，value 里写 pos/neg prompts
CONCEPTS: dict[str, dict[str, list[str]]] = {
    # ---------------------------------------------------------
    # 1. 基础物体类 (Object Existence) - 类似你的 Car
    # ---------------------------------------------------------
    "dog": {
        "pos": [
            "a photo of a dog on the grass",           # 场景1：草地狗
            "a dog sitting on a sofa",                 # 场景2：沙发狗
            "a puppy running in the park",             # 场景3：公园狗
            "a close-up portrait of a dog",            # 场景4：特写
            "a dog sleeping on a rug",                 # 场景5：地毯狗
        ],
        "neg": [
            "a photo of empty green grass",            # 场景1：空草地
            "an empty sofa",                           # 场景2：空沙发
            "an empty park with trees",                # 场景3：空公园
            "a close-up photo of a background texture",# 场景4：空背景
            "an empty rug on the floor",               # 场景5：空地毯
        ],
    },

    "car": {
        "pos": [
            "a photo of a car on the road",        # 场景1：路上有车
            "a car parked in a garage",            # 场景2：车库有车
            "a vehicle on the street",             # 场景3：街上有车
            "a car driving on the highway",        # 场景4：高速有车
            "an automobile in a parking lot",      # 场景5：停车场有车
            "a close-up photo of a car",           # 场景6：特写
            "a sedan in the city",                 # 场景7：城市背景
        ],
        "neg": [
            "a photo of an empty road",            # 场景1：空路 (去掉车)
            "an empty garage",                     # 场景2：空车库
            "an empty street",                     # 场景3：空街
            "an empty highway",                    # 场景4：空高速
            "an empty parking lot",                # 场景5：空停车场
            "a close-up photo of nothing",         # 场景6：虚空/背景
            "an empty city street",                # 场景7：无车的城市,
        ],
    },

    # ---------------------------------------------------------
    # 2. 颜色属性类 (Color Attribute) - 核心：同物体，异颜色
    # 用于验证解耦：能否只提取“红色”而不提取“苹果”
    # ---------------------------------------------------------
    "red": {
        "pos": [
            "a photo of a red apple",                  # 物体1
            "a shiny red car",                         # 物体2
            "a woman wearing a red dress",             # 物体3
            "a red rose in a vase",                    # 物体4
            "a red book on the table",                 # 物体5
        ],
        "neg": [
            "a photo of a green apple",                # 对照1：变绿
            "a shiny blue car",                        # 对照2：变蓝
            "a woman wearing a black dress",           # 对照3：变黑
            "a white rose in a vase",                  # 对照4：变白
            "a yellow book on the table",              # 对照5：变黄
        ],
    },

    # ---------------------------------------------------------
    # 3. 局部饰品类 (Local Accessory) - 核心：Safety/Erasure Proxy
    # 用于验证“无损擦除”：去掉眼镜，但眼睛还在
    # ---------------------------------------------------------
    "glasses": {
        "pos": [
            "a portrait of a man wearing glasses",     # 人物1
            "a woman with sunglasses on the beach",    # 人物2
            "a student wearing reading glasses",       # 人物3
            "an old man with spectacles",              # 人物4
            "a close-up of a face with eyewear",       # 人物5
        ],
        "neg": [
            "a portrait of a man without glasses",     # 对照1：无眼镜
            "a woman with bare face on the beach",     # 对照2：无墨镜
            "a student without glasses",               # 对照3：无眼镜
            "an old man without spectacles",           # 对照4：无眼镜
            "a close-up of a face without eyewear",    # 对照5：裸眼
        ],
    },

    # ---------------------------------------------------------
    # 4. 艺术风格类 (Artistic Style) - 核心：Global Style Transfer
    # 用于验证 Mid-Block 对整体纹理和笔触的控制
    # ---------------------------------------------------------
    "van_gogh": {
        "pos": [
            "a painting of a starry night in Van Gogh style",   # 场景1
            "a sunflower vase painted by Van Gogh",             # 场景2
            "an oil painting of a cypress tree, Van Gogh style",# 场景3
            "a portrait of a man, impressionist style",         # 场景4
            "a landscape painting with swirling clouds",        # 场景5
        ],
        "neg": [
            "a realistic photo of a starry night",              # 对照1：写实
            "a photo of a sunflower vase",                      # 对照2：写实
            "a photo of a cypress tree",                        # 对照3：写实
            "a realistic photo of a man",                       # 对照4：写实
            "a realistic landscape photo",                      # 对照5：写实
        ],
    },

    # ---------------------------------------------------------
    # 5. 抽象/环境类 (Atmosphere) - 核心：环境光与氛围
    # ---------------------------------------------------------
    "cyberpunk": {
        "pos": [
            "a cyberpunk city street at night",        # 场景1
            "a woman with neon lights, cyberpunk style",# 场景2
            "a futuristic car in a cyberpunk setting", # 场景3
            "a robot in a neon-lit alleyway",          # 场景4
        ],
        "neg": [
            "a normal city street at day",             # 对照1：普通白天
            "a woman in natural lighting",             # 对照2：自然光
            "a regular car on a normal street",        # 对照3：普通车
            "a person in a dark brick alleyway",       # 对照4：普通巷子
        ],
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run exp53 (TARIS) for concepts in a dict")
    ap.add_argument("--output_root", type=str, default="./out_taris_dict")
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
        help="只跑这些概念名（空表示全跑），例如 --only red_vs_blue glasses",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_root = os.path.expanduser(args.output_root)
    os.makedirs(output_root, exist_ok=True)

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

    only = set(map(str, args.only)) if args.only else None
    for concept_name, spec in CONCEPTS.items():
        if only is not None and concept_name not in only:
            continue
        pos = list(spec.get("pos", []))
        neg = list(spec.get("neg", []))
        if not pos or not neg:
            print(f"[skip] {concept_name}: pos/neg 为空")
            continue

        concept_cfg = ConceptLocateConfig(
            block=str(args.loc_block),
            concept_name=str(concept_name),
            pos_prompts=tuple(pos),
            neg_prompts=tuple(neg),
            t_start=int(args.taris_t_start),
            t_end=int(args.taris_t_end),
            num_t_samples=int(args.taris_num_steps),
            delta=float(args.taris_delta),
            top_k=int(args.taris_top_k),
        )

        run_exp53_concept_locator_taris(
            model_cfg,
            sae_cfg,
            run_cfg,
            concept_cfg,
            output_root,
            session=session,
        )


if __name__ == "__main__":
    main()
