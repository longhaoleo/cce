import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor


# -----------------------------
# 1) Token 索引定位（支持子词）
# -----------------------------
def find_subsequence_indices(full: List[int], sub: List[int]) -> List[int]:
    if len(sub) == 0:
        return []
    for i in range(len(full) - len(sub) + 1):
        if full[i : i + len(sub)] == sub:
            return list(range(i, i + len(sub)))
    return []


def get_token_indices_for_phrase(tokenizer, prompt: str, phrase: str) -> Tuple[List[int], Dict]:
    max_len = tokenizer.model_max_length
    enc_prompt = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prompt_ids = enc_prompt.input_ids[0].tolist()

    enc_phrase = tokenizer(phrase, add_special_tokens=False, return_tensors="pt")
    phrase_ids = enc_phrase.input_ids[0].tolist()

    indices = find_subsequence_indices(prompt_ids, phrase_ids)

    debug = {
        "prompt": prompt,
        "phrase": phrase,
        "prompt_tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
        "phrase_tokens": tokenizer.convert_ids_to_tokens(phrase_ids),
        "indices": indices,
    }
    return indices, debug


# -----------------------------
# 2) Head-wise cross-attn 捕获（可选 mid / up-block）
# -----------------------------
@dataclass
class HeadMapStore:
    maps: List[np.ndarray] = field(default_factory=list)  # each: (heads, H, W)

    def add(self, head_map: np.ndarray):
        self.maps.append(head_map)

    def mean(self) -> Optional[np.ndarray]:
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


# -----------------------------
# 3) Prompt embedding 编码（支持 mode B 的 embedding 置零）
# -----------------------------
def encode_prompt(pipe: StableDiffusionPipeline, prompt: str, device: torch.device, dtype: torch.dtype):
    tok = pipe.tokenizer
    max_len = tok.model_max_length

    text_in = tok(prompt, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = text_in.input_ids.to(device)
    attn_mask = text_in.attention_mask.to(device)

    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids, attention_mask=attn_mask)[0].to(dtype)

    neg_in = tok("", padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    neg_ids = neg_in.input_ids.to(device)
    neg_mask = neg_in.attention_mask.to(device)

    with torch.no_grad():
        negative_embeds = pipe.text_encoder(neg_ids, attention_mask=neg_mask)[0].to(dtype)

    return prompt_embeds, negative_embeds


def zero_out_token_embeddings(prompt_embeds: torch.Tensor, token_indices: List[int]) -> torch.Tensor:
    pe = prompt_embeds.clone()
    for idx in token_indices:
        if 0 <= idx < pe.shape[1]:
            pe[:, idx, :] = 0
    return pe


# -----------------------------
# 4) Causal Localization 核心：get_causal_mask
# -----------------------------
def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


def get_causal_mask(
    pipe: StableDiffusionPipeline,
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
    subj_idxs, subj_dbg = get_token_indices_for_phrase(pipe.tokenizer, prompt_clean, subj_word)
    obj_idxs, obj_dbg = get_token_indices_for_phrase(pipe.tokenizer, prompt_clean, obj_word)

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

    pe_clean, ne_clean = encode_prompt(pipe, prompt_clean, device=device, dtype=dtype)
    _ = pipe(
        prompt=None,
        prompt_embeds=pe_clean,
        negative_prompt_embeds=ne_clean,
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
        pe_pert, ne_pert = encode_prompt(pipe, prompt_pert, device=device, dtype=dtype)
    elif mode.upper() == "B":
        # 方式 B：保持 prompt 不变，但把 smoking 的 embedding 置零
        pe_pert, ne_pert = encode_prompt(pipe, prompt_clean, device=device, dtype=dtype)
        pe_pert = zero_out_token_embeddings(pe_pert, obj_idxs)
    else:
        raise ValueError("mode 必须是 'A' 或 'B'。")

    _ = pipe(
        prompt=None,
        prompt_embeds=pe_pert,
        negative_prompt_embeds=ne_pert,
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
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(1, 1, 1); ax.imshow(img); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    prompt = "Mickey Mouse smoking a cigar in a dark alley"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = os.path.expanduser("~/datasets/sd1.5")
    pipe = StableDiffusionPipeline.from_pretrained(
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

    configs = [
        ("mid", (16, 16)),
        ("up", (32, 32)),
        ("up", (64, 64)),
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
            height=512,
            width=512,
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
            height=512,
            width=512,
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

    print(f"Saved comparisons to: {out_dir}")


if __name__ == "__main__":
    main()
