"""
因果定位（Causal Localization）脚本：对比扰动前后注意力图并生成 mask。
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor

from model_utils import get_token_indices, encode_prompt_for_pipe, zero_out_token_embeddings, is_sdxl
# 输出 A/B 两种扰动下的注意力对比图，用于找到“哪块区域对移除 cigar 最敏感”，从而得到 mask。

# 它输出的内容（每个分辨率一组）：

# *_clean.png：不扰动时（正常 prompt）Mickey 对应的 cross-attn 热力图
# *_modeA_grid.png：Mode A（去词） 的 4 联图
# M_clean：正常 prompt 下 Mickey 的注意力
# M_perturbed：删掉 cigar 后 Mickey 的注意力
# Diff：两者差值（归一化）
# Mask：阈值化后的二值 mask
# *_modeB_grid.png：Mode B（embedding 置零） 的 4 联图
# *_diff_modeA.png / *_diff_modeB.png：A/B 的差值图，方便横向对比
# clean_image.png：一张正常生成的图，仅作视觉参考


# 已移至 attn_utils.get_token_indices_for_phrase


# -----------------------------
# 2) Head-wise cross-attn 捕获（可选 mid / up-block）
# -----------------------------
@dataclass
class HeadMapStore:
    maps: List[np.ndarray] = field(default_factory=list)  # each: (heads, H, W)

    def add(self, head_map: np.ndarray):
        """追加一张 head-wise 注意力图。"""
        self.maps.append(head_map)

    def mean(self) -> Optional[np.ndarray]:
        """返回所有 head-wise 图的平均结果。"""
        if len(self.maps) == 0:
            return None
        return np.mean(np.stack(self.maps, axis=0), axis=0)  # (heads, H, W)


class BlockCrossAttnCaptureProcessor(AttnProcessor):
    """
    只用于指定 block 的 attn2（cross-attn）。
    记录 Mickey 对应的 attention map（按 token span 求和），保留 head 维度。
    """
    def __init__(
        self,
        token_indices: List[int],
        store: HeadMapStore,
        target_hw: Tuple[int, int] = (16, 16),
        capture_only_cond: bool = True,
    ):
        super().__init__()
        self.token_indices = token_indices
        self.store = store
        self.target_hw = target_hw
        self.capture_only_cond = capture_only_cond

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
        """执行 attention 计算并抓取 mid-block cross-attn。"""
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            height = width = None

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) * scale

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            # self-attn，不采集
            encoder_hidden_states = hidden_states
        else:
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # [B*H, Q, K]

        # 只采 cross-attn（attn2）
        if is_cross:
            heads = attn.heads
            bh, q_len, k_len = attention_probs.shape
            bsz = bh // heads
            attn_4d = attention_probs.view(bsz, heads, q_len, k_len)  # [B, heads, Q, K]

            # CFG 情况：B=2（uncond+cond），只保留 cond
            if self.capture_only_cond and (bsz % 2 == 0) and (bsz >= 2):
                attn_4d = attn_4d[bsz // 2 :]  # [cond_B, heads, Q, K]

            # 平均 batch（通常 cond_B=1） -> [heads, Q, K]
            attn_hqk = attn_4d.mean(dim=0)

            side = int(math.sqrt(q_len))
            if side * side == q_len:
                H = W = side
                if (H, W) == self.target_hw and len(self.token_indices) > 0:
                    # 对 token span 求和： [heads, Q, K] -> [heads, Q]
                    vec = attn_hqk[:, :, self.token_indices].sum(dim=2)
                    head_map = vec.view(heads, H, W).detach().float().cpu().numpy()
                    self.store.add(head_map)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", True):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


# encode_prompt / zero_out_token_embeddings 已移至 model_utils


# -----------------------------
# 4) Causal Localization 核心：get_causal_mask
# -----------------------------
def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """对 2D 数组做 min-max 归一化。"""
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


def get_causal_mask(
    pipe: DiffusionPipeline,
    prompt_clean: str,
    subj_word: str = "Mickey",
    obj_word: str = "smoking",
    mode: str = "A",            # "A": 去词；"B": embedding 置零
    tau: float = 0.2,
    seed: int = 1234,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 10,
    guidance_scale: float = 7.5,
    target_hw: Tuple[int, int] = (16, 16),
    block_scope: Literal["mid", "up"] = "mid",
):
    """两次前向（clean/perturbed）生成 Diff 与二值 Mask。"""
    device = pipe.device
    dtype = pipe.unet.dtype

    # 1) 固定种子 + 固定初始噪声 latents，确保两次 pass 只差在条件输入
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    latent_h, latent_w = height // 8, width // 8
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=gen,
        device=device,
        dtype=dtype,
    )

    # 2) token 索引（严格基于 tokenizer）
    subj_idxs, subj_dbg = get_token_indices(pipe, prompt_clean, subj_word)
    obj_idxs, obj_dbg = get_token_indices(pipe, prompt_clean, obj_word)

    if not subj_idxs:
        raise ValueError(f"找不到主体词 {subj_word} 在 prompt 中的 token span。")
    if not obj_idxs:
        raise ValueError(f"找不到客体词 {obj_word} 在 prompt 中的 token span。")

    # 保存原 processors，便于恢复
    orig_procs = dict(pipe.unet.attn_processors)

    def patch_block_attn2(store: HeadMapStore):
        patched = {}
        for name, proc in orig_procs.items():
            is_mid = name.startswith("mid_block")
            is_up = name.startswith("up_blocks")
            is_attn2 = ".attn2." in name
            is_target = (block_scope == "mid" and is_mid) or (block_scope == "up" and is_up)
            if is_target and is_attn2:
                patched[name] = BlockCrossAttnCaptureProcessor(
                    token_indices=subj_idxs,
                    store=store,
                    target_hw=target_hw,
                    capture_only_cond=True,
                )
            else:
                patched[name] = proc
        pipe.unet.set_attn_processor(patched)

    # 3) Pass 1 (Clean)
    clean_store = HeadMapStore()
    patch_block_attn2(clean_store)

    enc_clean = encode_prompt_for_pipe(pipe, prompt_clean, device=device, dtype=dtype)
    _ = pipe(
        prompt=None,
        **enc_clean,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        latents=latents.clone(),
        output_type="latent",      # 不做 VAE decode，快一些；attention 捕获不受影响
    )

    M_clean_heads = clean_store.mean()
    if M_clean_heads is None:
        raise RuntimeError("没有捕获到目标分辨率的 cross-attn。请确认 height/width 与 block_scope 设置是否匹配。")

    # 4) Pass 2 (Perturbed)
    pert_store = HeadMapStore()
    patch_block_attn2(pert_store)

    if mode.upper() == "A":
        # 方式 A：直接去掉 smoking
        prompt_pert = " ".join([w for w in prompt_clean.split() if w.lower() != obj_word.lower()])
        enc_pert = encode_prompt_for_pipe(pipe, prompt_pert, device=device, dtype=dtype)
    elif mode.upper() == "B":
        # 方式 B：保持 prompt 不变，但把 smoking 的 embedding 置零
        if is_sdxl(pipe):
            # SDXL 下也可以 zero，但默认仍建议用 A（更稳定）
            enc_pert = encode_prompt_for_pipe(pipe, prompt_clean, device=device, dtype=dtype)
        else:
            enc_pert = encode_prompt_for_pipe(pipe, prompt_clean, device=device, dtype=dtype)
        enc_pert["prompt_embeds"] = zero_out_token_embeddings(enc_pert["prompt_embeds"], obj_idxs)
    else:
        raise ValueError("mode 必须是 'A' 或 'B'。")

    _ = pipe(
        prompt=None,
        **enc_pert,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        latents=latents.clone(),
        output_type="latent",
    )

    M_pert_heads = pert_store.mean()
    if M_pert_heads is None:
        raise RuntimeError("Perturbed pass 没有捕获到目标分辨率的 cross-attn。")

    # 恢复 processors
    pipe.unet.set_attn_processor(orig_procs)

    # 5) Diff：先对 8 个 heads 平均，再做 |.|，再 min-max normalize，再阈值化
    M_clean = M_clean_heads.mean(axis=0)   # (16,16)
    M_pert  = M_pert_heads.mean(axis=0)    # (16,16)

    Diff = np.abs(M_clean - M_pert)
    Diff_norm = minmax_normalize(Diff)
    Mask = (Diff_norm > tau).astype(np.float32)

    debug = {
        "subj_debug": subj_dbg,
        "obj_debug": obj_dbg,
        "mode": mode,
        "tau": tau,
        "target_hw": target_hw,
        "block_scope": block_scope,
    }
    return M_clean, M_pert, Diff_norm, Mask, debug


# -----------------------------
# 5) 可视化
# -----------------------------
def save_grid_four(path: str, title: str, M_clean: np.ndarray, M_pert: np.ndarray, Diff: np.ndarray, Mask: np.ndarray):
    """保存 4 联图（clean / perturbed / diff / mask）。"""
    fig = plt.figure(figsize=(10, 9))
    ax1 = plt.subplot(2, 2, 1); ax1.imshow(M_clean); ax1.set_title("M_clean (Mickey)"); ax1.axis("off")
    ax2 = plt.subplot(2, 2, 2); ax2.imshow(M_pert);  ax2.set_title("M_perturbed (Mickey)"); ax2.axis("off")
    ax3 = plt.subplot(2, 2, 3); ax3.imshow(Diff);    ax3.set_title("Diff (abs, min-max)"); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.imshow(Mask);    ax4.set_title("Mask (tau=0.2)"); ax4.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_single(path: str, title: str, img: np.ndarray):
    """保存单张热力图到磁盘。"""
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(1, 1, 1); ax.imshow(img); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """主流程：生成 A/B 两种扰动对比图，并保存 clean 图。"""
    prompt = "Mickey Mouse smoking a cigar in a dark alley"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = 512, 512
    seed = 1234

    model_id = os.path.expanduser("~/datasets/sd1.5")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",      
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    pipe.enable_attention_slicing()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "causal_compare")
    os.makedirs(out_dir, exist_ok=True)

    # mid 分辨率 = H/64, W/64；up-block 分辨率依次为 H/32、H/16、H/8
    mid_hw = (height // 64, width // 64)
    up_hw_32 = (height // 16, width // 16)
    up_hw_64 = (height // 8, width // 8)

    configs = [
        ("mid", mid_hw),
        ("up", up_hw_32),
        ("up", up_hw_64),
    ]

    for block_scope, target_hw in configs:
        tag = f"{block_scope}_{target_hw[0]}x{target_hw[1]}"

        # Mode A: remove object word
        M_clean_A, M_pert_A, Diff_A, Mask_A, _ = get_causal_mask(
            pipe,
            prompt_clean=prompt,
            subj_word="Mickey",
            obj_word="cigar",
            mode="A",
            tau=0.2,
            seed=1234,
            height=height,
            width=width,
            num_inference_steps=10,
            guidance_scale=7.5,
            target_hw=target_hw,
            block_scope=block_scope,
        )

        # Mode B: zero object embedding
        M_clean_B, M_pert_B, Diff_B, Mask_B, _ = get_causal_mask(
            pipe,
            prompt_clean=prompt,
            subj_word="Mickey",
            obj_word="cigar",
            mode="B",
            tau=0.2,
            seed=1234,
            height=height,
            width=width,
            num_inference_steps=10,
            guidance_scale=7.5,
            target_hw=target_hw,
            block_scope=block_scope,
        )

        # 保存单图：不扰动（clean）
        save_single(os.path.join(out_dir, f"{tag}_clean.png"), f"Clean @ {tag}", M_clean_A)

        # 保存 A/B 结果对比（4联图）
        save_grid_four(
            os.path.join(out_dir, f"{tag}_modeA_grid.png"),
            f"Mode A (remove 'cigar') @ {tag}",
            M_clean_A,
            M_pert_A,
            Diff_A,
            Mask_A,
        )
        save_grid_four(
            os.path.join(out_dir, f"{tag}_modeB_grid.png"),
            f"Mode B (zero 'cigar') @ {tag}",
            M_clean_B,
            M_pert_B,
            Diff_B,
            Mask_B,
        )

        # 保存差异图，方便 A/B 直接对比
        save_single(os.path.join(out_dir, f"{tag}_diff_modeA.png"), f"Diff A @ {tag}", Diff_A)
        save_single(os.path.join(out_dir, f"{tag}_diff_modeB.png"), f"Diff B @ {tag}", Diff_B)

    # # 额外保存一张 clean 图，方便对照
    # generator = torch.Generator(device=device).manual_seed(seed)
    # with torch.autocast(device_type=device, enabled=(device == "cuda")):
    #     result = pipe(
    #         prompt=prompt,
    #         height=height,
    #         width=width,
    #         num_inference_steps=30,
    #         guidance_scale=7.5,
    #         generator=generator,
    #     )
    # clean_path = os.path.join(out_dir, "clean_image.png")
    # result.images[0].save(clean_path)

    # print(f"Saved clean image to: {clean_path}")
    # print(f"Saved comparisons to: {out_dir}")


if __name__ == "__main__":
    main()
