"""
生成图像并抓取 cross-attention 热力图的示例脚本（适配 SD/SDXL）。
"""

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor

from attn_utils import save_heatmap_exact_resolution, to_colormap_uint8
from model_utils import get_token_indices


# -----------------------------
# 1) Token 索引定位（支持子词）
# -----------------------------
# 已移至 attn_utils.get_token_indices_for_phrase


# -----------------------------
# 2) Attention Map 存储与导出
# -----------------------------
@dataclass
class AttnMapStore:
    """
    按 token_name 与分辨率保存多次采样到的 attention map，最后做平均。
    data[token_name][(H,W)] -> list[np.ndarray(H,W)]
    """
    data: Dict[str, Dict[Tuple[int, int], List[np.ndarray]]] = field(default_factory=dict)

    def add(self, token_name: str, attn_2d: np.ndarray):
        """追加一次注意力图。"""
        h, w = attn_2d.shape
        self.data.setdefault(token_name, {})
        self.data[token_name].setdefault((h, w), [])
        self.data[token_name][(h, w)].append(attn_2d)

    def mean_maps(self) -> Dict[str, Dict[Tuple[int, int], np.ndarray]]:
        """对同一分辨率的多张 map 做平均。"""
        out = {}
        for token_name, by_res in self.data.items():
            out[token_name] = {}
            for res, maps in by_res.items():
                if len(maps) == 0:
                    continue
                out[token_name][res] = np.mean(np.stack(maps, axis=0), axis=0)
        return out


# 已移至 attn_utils.to_colormap_uint8 / save_heatmap_exact_resolution


# -----------------------------
# 3) 自定义 AttnProcessor：只抓 Cross-Attn（attn2）
#    且只用于 mid_block / up_blocks
# -----------------------------
class CrossAttnCaptureProcessor(AttnProcessor):
    def __init__(
        self,
        layer_name: str,
        store: AttnMapStore,
        token_groups: Dict[str, List[int]],
        capture_only_cond: bool = True,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.store = store
        self.token_groups = token_groups
        self.capture_only_cond = capture_only_cond

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """执行标准 attention 计算并抓取 cross-attn。"""
        # ----------- 标准 AttnProcessor 逻辑（兼容 SD1.5）-----------
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
            # self-attn：直接走原逻辑，但我们不采集
            encoder_hidden_states = hidden_states
        else:
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ----------- 只采集 cross-attn，并且仅取 conditional 分支 -----------
        # attention_probs: [batch*heads, Q, K]
        if is_cross:
            heads = attn.heads
            bh, q_len, k_len = attention_probs.shape
            bsz = bh // heads
            attn_4d = attention_probs.view(bsz, heads, q_len, k_len)

            # classifier-free guidance 时，bsz 往往是 2（uncond + cond）
            if self.capture_only_cond and (bsz % 2 == 0) and (bsz >= 2):
                attn_4d = attn_4d[bsz // 2 :]

            # 平均 heads -> [B, Q, K]，再平均 batch -> [Q, K]
            attn_qk = attn_4d.mean(dim=1).mean(dim=0)

            # Q = H*W（通常是图像token数），尝试还原二维
            side = int(math.sqrt(q_len))
            if side * side == q_len:
                H = W = side
                # 对每个 token_group 抽列并求和（适配子词）
                for name, idxs in self.token_groups.items():
                    if not idxs:
                        continue
                    # [Q, K] -> 取 K 维的若干 token 列并求和 -> [Q]
                    vec_q = attn_qk[:, idxs].sum(dim=1)
                    attn_2d = vec_q.view(H, W).detach().float().cpu().numpy()
                    self.store.add(name, attn_2d)
            # 如果 q_len 不是完全平方（极少数情况），就不保存，避免误导

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
# 4) 主流程
# -----------------------------
def main():
    """主流程：生成图像并保存各层 attention 热力图。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = os.path.expanduser("~/datasets/sd1.5")
    prompt = "Mickey Mouse smoking a cigar in a dark alley"
    # prompt = "A Muslim eating pork at home."
    # prompt = "taylor swift without clothes"

    out_dir = "../output/mikey_cigar_crossattn_maps"
    os.makedirs(out_dir, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",      
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    # 建议：开启注意力切片节省显存（可选）
    pipe.enable_attention_slicing()

    # 1) 找 token 索引（严格基于 tokenizer）
    mickey_idxs, mickey_dbg = get_token_indices(pipe, prompt, "Mickey")
    smoking_idxs, smoking_dbg = get_token_indices(pipe, prompt, "cigar")

    token_groups = {
        "Mickey": mickey_idxs,
        "cigar": smoking_idxs,
    }

    # 2) 给 mid_block / up_blocks 的 attn2 注册捕获 processor
    store = AttnMapStore()
    attn_procs = pipe.unet.attn_processors  # Dict[name -> processor]

    new_procs = {}
    for name, proc in attn_procs.items():
        # name 示例（不同 diffusers 版本略有差异）：
        # "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
        # "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor"
        # 我们只要 mid_block 或 up_blocks，并且只要 attn2（cross-attn）
        is_mid_or_up = name.startswith("mid_block") or name.startswith("up_blocks")
        is_cross_attn = ".attn2." in name

        if is_mid_or_up and is_cross_attn:
            new_procs[name] = CrossAttnCaptureProcessor(
                layer_name=name,
                store=store,
                token_groups=token_groups,
                capture_only_cond=True,
            )
        else:
            # 其余保持原样（包括 down_blocks，及 attn1 self-attn）
            new_procs[name] = proc

    pipe.unet.set_attn_processor(new_procs)

    # 3) 一次推理生成
    generator = torch.Generator(device=device).manual_seed(1234)

    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        result = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        )

    image = result.images[0]
    image_path = os.path.join(out_dir, "generated.png")
    image.save(image_path)
    print(f"Saved generated image to: {image_path}")

    # 4) 汇总并按分辨率保存热力图
    mean_maps = store.mean_maps()
    for token_name, by_res in mean_maps.items():
        for (h, w), attn_2d in by_res.items():
            # 保存严格 (h,w) 分辨率的热力图
            heat_path = os.path.join(out_dir, f"crossattn_{token_name}_{h}x{w}.png")
            save_heatmap_exact_resolution(attn_2d, heat_path)
            print(f"Saved heatmap: {heat_path}")

            # 可选：额外保存一份放大版（便于肉眼看），不影响“原始分辨率版”的要求
            up = Image.fromarray(to_colormap_uint8(attn_2d), mode="RGB").resize((w * 16, h * 16), resample=Image.NEAREST)
            up_path = os.path.join(out_dir, f"crossattn_{token_name}_{h}x{w}_x16.png")
            up.save(up_path)
            print(f"Saved upscaled heatmap: {up_path}")

    print("Done.")


if __name__ == "__main__":
    main()
