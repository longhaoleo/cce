"""
两阶段手术示例脚本：先定位 mask，再执行 attention 手术生成对比图。
"""

import os
from typing import Tuple
import torch
from PIL import Image

from diffusers import DiffusionPipeline

from test_indentifier_sd15 import get_causal_mask
from attn_surgery_processor import attach_surgery_processors
from attn_utils import save_heatmap, save_mask
from mask_utils import resize_mask_to
from model_utils import get_token_indices


# to_colormap_uint8 / save_heatmap / save_mask 已移至 attn_utils


def save_side_by_side(path: str, left: Image.Image, right: Image.Image):
    """保存左右拼接对比图。"""
    w = left.width + right.width
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(path)


def main():
    """主流程：生成 baseline / surgery 图，并保存对比结果。"""
    # -------- 配置区 --------
    # prompt = "Mickey Mouse smoking a cigar in a dark alley"
    # subject_word = "Mickey"
    # object_word = "cigar"
    # prompt = "A Muslim eating pork at home."
    # subject_word = "Muslim"
    # object_word = "pork"
    prompt = "A kid handle a knife."
    subject_word = "kid"
    object_word = "knife"
    
    height, width = 512, 512
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = 1234

    # 第一阶段：用前 10 步定位（mid 分辨率 = latent 再下采样 8 倍 = H/64, W/64）
    num_steps_mask = 10
    tau = 0.1
    mask_mode = "B"  # "A": 去词；"B": embedding 置零

    model_id = os.path.expanduser("~/datasets/sd1.5")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "surgery_compare")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    pipe.enable_attention_slicing()

    # -------- 1) 计算 surgery_mask（定位阶段）--------
    mid_hw = (height // 64, width // 64)
    M_clean, M_pert, Diff_norm, Mask, dbg = get_causal_mask(
        pipe,
        prompt_clean=prompt,
        subj_word=subject_word,
        obj_word=object_word,
        mode=mask_mode,
        tau=tau,
        seed=seed,
        height=height,
        width=width,
        num_inference_steps=num_steps_mask,
        guidance_scale=guidance_scale,
        target_hw=mid_hw,
        block_scope="mid",
    )

    # 保存定位阶段的可视化
    save_heatmap(os.path.join(out_dir, "mask_M_clean.png"), M_clean)
    save_heatmap(os.path.join(out_dir, "mask_M_pert.png"), M_pert)
    save_heatmap(os.path.join(out_dir, "mask_Diff.png"), Diff_norm)
    save_mask(os.path.join(out_dir, "mask_Binary.png"), Mask)

    # -------- 2) 生成基线图（不扰动）--------
    gen_base = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        base = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen_base,
        ).images[0]
    base_path = os.path.join(out_dir, "before_surgery.png")
    base.save(base_path)

    # -------- 3) 挂载 surgery processor 并生成对照图 --------
    subj_idxs, _ = get_token_indices(pipe, prompt, subject_word)
    obj_idxs, _ = get_token_indices(pipe, prompt, object_word)
    if not subj_idxs or not obj_idxs:
        raise ValueError("主体或客体 token 索引为空，请确认 prompt 与词语。")

    # 统一走 mask_utils，确保尺寸一致
    Mask = resize_mask_to(Mask, mid_hw, mode="nearest")
    mask_tensor = torch.from_numpy(Mask).to(device=device, dtype=pipe.unet.dtype)
    attach_surgery_processors(
        pipe,
        subject_token_indices=subj_idxs,
        object_token_indices=obj_idxs,
        surgery_mask=mask_tensor,
        apply_to_blocks=("mid", "up"),
        apply_to_cond_only=True,
        renorm=True,
    )

    gen_surgery = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        after = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen_surgery,
        ).images[0]
    after_path = os.path.join(out_dir, "after_surgery.png")
    after.save(after_path)

    # -------- 4) 保存对比图 --------
    compare_path = os.path.join(out_dir, "compare_before_after.png")
    save_side_by_side(compare_path, base, after)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
