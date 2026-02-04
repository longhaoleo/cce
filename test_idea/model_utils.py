"""
模型相关的通用工具函数：SD/SDXL 兼容、token 索引与 prompt 编码。
"""

from typing import Dict, List, Tuple

import torch

from attn_utils import get_token_indices_for_phrase


def is_sdxl(pipe) -> bool:
    """
    判断是否为 SDXL Pipeline（有 tokenizer_2 或 text_encoder_2）。
    """
    return hasattr(pipe, "tokenizer_2") or hasattr(pipe, "text_encoder_2")


def get_token_indices(pipe, prompt: str, phrase: str) -> Tuple[List[int], Dict]:
    """
    返回在“拼接后的 prompt_embeds 序列”中的 token 索引。
    - SD1.5：只使用 tokenizer
    - SDXL：tokenizer + tokenizer_2，第二段索引会自动加偏移
    """
    tokenizers = [pipe.tokenizer]
    if hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2 is not None:
        tokenizers.append(pipe.tokenizer_2)

    indices: List[int] = []
    debug: Dict = {}
    offset = 0
    for i, tok in enumerate(tokenizers):
        max_len = tok.model_max_length
        idxs, dbg = get_token_indices_for_phrase(tok, prompt, phrase, max_length=max_len)
        indices.extend([j + offset for j in idxs])
        debug[f"tok{i}"] = dbg
        offset += max_len
    return indices, debug


def encode_prompt_for_pipe(
    pipe,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    negative_prompt: str = "",
):
    """
    统一封装 prompt 编码，兼容 SD / SDXL。
    返回一个 dict，供 pipe 调用。
    """
    if is_sdxl(pipe):
        prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        return {
            "prompt_embeds": prompt_embeds.to(dtype),
            "negative_prompt_embeds": neg_embeds.to(dtype),
            "pooled_prompt_embeds": pooled.to(dtype),
            "negative_pooled_prompt_embeds": neg_pooled.to(dtype),
        }

    # SD1.x：单 encoder
    tok = pipe.tokenizer
    max_len = tok.model_max_length
    text_in = tok(prompt, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = text_in.input_ids.to(device)
    attn_mask = text_in.attention_mask.to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids, attention_mask=attn_mask)[0].to(dtype)

    neg_in = tok(negative_prompt, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    neg_ids = neg_in.input_ids.to(device)
    neg_mask = neg_in.attention_mask.to(device)
    with torch.no_grad():
        negative_embeds = pipe.text_encoder(neg_ids, attention_mask=neg_mask)[0].to(dtype)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
    }


def zero_out_token_embeddings(prompt_embeds: torch.Tensor, token_indices: List[int]) -> torch.Tensor:
    """
    将 prompt_embeds 中指定 token 位置置零。
    兼容 SDXL（token_indices 已含偏移）。
    """
    pe = prompt_embeds.clone()
    for idx in token_indices:
        if 0 <= idx < pe.shape[1]:
            pe[:, idx, :] = 0
    return pe
