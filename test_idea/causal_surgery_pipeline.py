"""
整合版 Causal Surgery Pipeline：定位 -> 手术 -> 迭代封锁 -> 生成对比图。
"""

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor

from attn_utils import save_heatmap, save_mask
from mask_utils import resize_mask_to, merge_masks
from model_utils import get_token_indices
from test_indentifier_sd15 import get_causal_mask
from attn_surgery_processor import attach_surgery_processors, CaptureMapStore


# -----------------------------
# 1) 通用：捕获指定 token 的 cross-attn map
# -----------------------------
@dataclass
class TokenMapStore:
    maps: List[np.ndarray] = field(default_factory=list)  # each: (H, W)

    def add(self, m: np.ndarray):
        """追加一张 2D attention map。"""
        self.maps.append(m)

    def mean(self) -> Optional[np.ndarray]:
        """返回所有 map 的平均结果。"""
        if len(self.maps) == 0:
            return None
        return np.mean(np.stack(self.maps, axis=0), axis=0)


class TokenCrossAttnCaptureProcessor(AttnProcessor):
    """
    只用于指定 block 的 attn2（cross-attn）。
    捕获指定 token span 的注意力图（按 token span 求和）。
    """
    def __init__(
        self,
        token_indices: List[int],
        store: TokenMapStore,
        target_hw: Tuple[int, int],
        capture_only_cond: bool = True,
    ):
        super().__init__()
        self.token_indices = token_indices
        self.store = store
        self.target_hw = target_hw
        self.capture_only_cond = capture_only_cond

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
        """执行 attention 计算并捕获指定 token 的 cross-attn。"""
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

        if is_cross and self.token_indices:
            heads = attn.heads
            bh, q_len, k_len = attention_probs.shape
            bsz = bh // heads
            attn_4d = attention_probs.view(bsz, heads, q_len, k_len)

            if self.capture_only_cond and (bsz % 2 == 0) and (bsz >= 2):
                attn_4d = attn_4d[bsz // 2 :]

            attn_hqk = attn_4d.mean(dim=0)  # [heads, Q, K]
            side = int(math.sqrt(q_len))
            if side * side == q_len:
                H = W = side
                if (H, W) == self.target_hw:
                    vec = attn_hqk[:, :, self.token_indices].sum(dim=2)  # [heads, Q]
                    m = vec.mean(dim=0).view(H, W).detach().float().cpu().numpy()
                    self.store.add(m)

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
# 2) 主流程封装：Causal Surgery Pipeline
# -----------------------------
class CausalSurgeryPipeline:
    def __init__(self, model_id: str, device: Optional[str] = None):
        """初始化模型与推理管线。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
            local_files_only=True,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)
        self.pipe.enable_attention_slicing()

    def _target_hw(self, height: int, width: int, block_scope: str) -> Tuple[int, int]:
        """根据 block 选择目标分辨率。"""
        # mid: H/64; up: 推荐用 H/16 或 H/8
        if block_scope == "mid":
            return (height // 64, width // 64)
        if block_scope == "up32":
            return (height // 16, width // 16)
        if block_scope == "up64":
            return (height // 8, width // 8)
        raise ValueError("block_scope must be one of: mid, up32, up64")

    def _capture_token_map(
        self,
        prompt: str,
        token_indices: List[int],
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        block_scope: str,
    ) -> np.ndarray:
        """在指定 block 上捕获 token attention map。"""
        target_hw = self._target_hw(height, width, block_scope)
        store = TokenMapStore()

        # 仅 patch 指定 block 的 attn2
        orig_procs = dict(self.pipe.unet.attn_processors)
        patched = {}
        for name, proc in orig_procs.items():
            is_attn2 = ".attn2." in name
            is_mid = name.startswith("mid_block")
            is_up = name.startswith("up_blocks")
            use = (block_scope == "mid" and is_mid) or (block_scope != "mid" and is_up)
            if is_attn2 and use:
                patched[name] = TokenCrossAttnCaptureProcessor(
                    token_indices=token_indices,
                    store=store,
                    target_hw=target_hw,
                    capture_only_cond=True,
                )
            else:
                patched[name] = proc

        self.pipe.unet.set_attn_processor(patched)

        gen = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            _ = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen,
                output_type="latent",
            )

        self.pipe.unet.set_attn_processor(orig_procs)

        m = store.mean()
        if m is None:
            raise RuntimeError("未捕获到目标分辨率的 cross-attn map。")
        return m

    def _next_block_mask(
        self,
        prompt: str,
        action_token: str,
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        threshold: float,
    ) -> Optional[np.ndarray]:
        """
        简化版“迭代封锁”：
        在 up-block 32x32 上抓 action 注意力图，直接把高响应区域加入 mask。
        """
        block_scope = "up32"
        action_idxs, _ = get_token_indices(self.pipe, prompt, action_token)
        if not action_idxs:
            return None

        action_map = self._capture_token_map(
            prompt, action_idxs, height, width, steps, guidance_scale, seed, block_scope
        )

        block_mask = (action_map > threshold).astype(np.float32)
        if block_mask.sum() < 1:
            return None
        return block_mask

    def generate(
        self,
        prompt: str,
        subject_token: str,
        action_token: str,
        threshold: float = 0.1,
        surgery_steps: int = 10,
        iterations: int = 2,
        action_mask_scope: str = "up32",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 1234,
        out_dir: Optional[str] = None,
    ):
        """
        统一入口：
        - 先做因果定位，得到 surgery_mask
        - 再挂载 processor 做正式生成
        - 支持前半程手术（surgery_steps），与迭代式封锁（默认 2 轮）
        """
        out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "output", "surgery_iter")
        os.makedirs(out_dir, exist_ok=True)

        # 轮次叠加 mask
        combined_mask = None
        # diff 的作用域与 action_mask_scope 对齐
        diff_scope = action_mask_scope
        base_hw = self._target_hw(height, width, diff_scope)
        diff_block_scope = "mid" if diff_scope == "mid" else "up"
        subj_idxs, _ = get_token_indices(self.pipe, prompt, subject_token)
        obj_idxs, _ = get_token_indices(self.pipe, prompt, action_token)

        for it in range(iterations):
            # 1) 通过因果扰动获得 mask（按 action_mask_scope 对齐）
            M_clean, M_pert, Diff_norm, Mask, _ = get_causal_mask(
                self.pipe,
                prompt_clean=prompt,
                subj_word=subject_token,
                obj_word=action_token,
                mode="B",
                tau=threshold,
                seed=seed,
                height=height,
                width=width,
                num_inference_steps=surgery_steps,
                guidance_scale=guidance_scale,
                target_hw=base_hw,
                block_scope=diff_block_scope,
            )

            if combined_mask is None:
                combined_mask = Mask
            else:
                combined_mask = merge_masks(combined_mask, Mask)

            # 1.5) 把 B（action）原始位置也加入 mask（可选 mid/up32/up64）
            action_idxs, _ = get_token_indices(self.pipe, prompt, action_token)
            if action_idxs:
                action_map_mid = self._capture_token_map(
                    prompt, action_idxs, height, width, surgery_steps, guidance_scale, seed, action_mask_scope
                )
                action_mask_mid = (action_map_mid > threshold).astype(np.float32)
                action_mask_mid = resize_mask_to(action_mask_mid, base_hw, mode="nearest")
                combined_mask = merge_masks(combined_mask, action_mask_mid)

            # 2) 迭代式封锁：抓 action 高响应区域，直接加入 mask
            extra_mask = self._next_block_mask(
                prompt=prompt,
                action_token=action_token,
                height=height,
                width=width,
                steps=surgery_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                threshold=threshold,
            )
            if extra_mask is not None:
                extra_mask = resize_mask_to(extra_mask, base_hw, mode="nearest")
                combined_mask = merge_masks(combined_mask, extra_mask)

            # 保存每轮 mask 可视化
            save_heatmap(os.path.join(out_dir, f"iter{it+1}_diff.png"), Diff_norm)
            save_mask(os.path.join(out_dir, f"iter{it+1}_mask.png"), combined_mask)

            # 2.5) 将当前 mask 挂载到 attention，再捕获 action 的新响应区域（闭环迭代）
            # 只做短步数，观察 “封锁后” 的迁移位置
            if obj_idxs:
                cap_store = CaptureMapStore()
                mask_tensor = torch.from_numpy(combined_mask).to(device=self.device, dtype=self.pipe.unet.dtype)
                orig_procs = attach_surgery_processors(
                    self.pipe,
                    subject_token_indices=subj_idxs,
                    object_token_indices=obj_idxs,
                    surgery_mask=mask_tensor,
                    apply_to_blocks=("mid", "up"),
                    apply_to_cond_only=True,
                    renorm=True,
                    capture_token_indices=obj_idxs,
                    capture_store=cap_store,
                    capture_target_hw=base_hw,
                    capture_only_cond=True,
                )

                gen_tmp = torch.Generator(device=self.device).manual_seed(seed)
                with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                    _ = self.pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=surgery_steps,
                        guidance_scale=guidance_scale,
                        generator=gen_tmp,
                        output_type="latent",
                    )

                self.pipe.unet.set_attn_processor(orig_procs)

                migrated = cap_store.mean()
                if migrated is not None:
                    migrated_mask = (migrated > threshold).astype(np.float32)
                    migrated_mask = resize_mask_to(migrated_mask, base_hw, mode="nearest")
                    combined_mask = merge_masks(combined_mask, migrated_mask)

        # 3) 生成原始图（不做手术），便于对比
        gen_base = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            base = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen_base,
            )
        base_path = os.path.join(out_dir, "baseline.png")
        base.images[0].save(base_path)

        # 4) 挂载 surgery processor
        subj_idxs, _ = get_token_indices(self.pipe, prompt, subject_token)
        obj_idxs, _ = get_token_indices(self.pipe, prompt, action_token)
        if not subj_idxs or not obj_idxs:
            raise ValueError("主体或客体 token 索引为空，请确认 prompt 与词语。")

        mask_tensor = torch.from_numpy(combined_mask).to(device=self.device, dtype=self.pipe.unet.dtype)
        orig_procs = attach_surgery_processors(
            self.pipe,
            subject_token_indices=subj_idxs,
            object_token_indices=obj_idxs,
            surgery_mask=mask_tensor,
            apply_to_blocks=("mid", "up"),
            apply_to_cond_only=True,
            renorm=True,
        )

        # 5) 只在前 surgery_steps 生效：通过 callback 切回原 processor
        state = {"switched": False}

        def _callback(step: int, timestep: int, latents):
            if (step + 1) >= surgery_steps and not state["switched"]:
                self.pipe.unet.set_attn_processor(orig_procs)
                state["switched"] = True

        gen = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                callback=_callback,
                callback_steps=1,
            )

        out_path = os.path.join(out_dir, "final.png")
        result.images[0].save(out_path)
        return {"baseline": base_path, "surgery": out_path}


if __name__ == "__main__":
    # 示例用法
    model_id = os.path.expanduser("~/datasets/sd1.5")
    pipe = CausalSurgeryPipeline(model_id=model_id)
    out = pipe.generate(
        # prompt="Mickey Mouse smoking a cigar in a dark alley",
        # subject_token="Mickey",
        # action_token="cigar",
        # prompt = "The child is holding a knife in the kitchen.",
        # subject_token = "kid",
        # action_token = "knife",
        prompt = "A strong man with black bat.",
        subject_token = "man",
        action_token = "bat",
        threshold=0.2,
        surgery_steps=5,
        iterations=2,
        out_dir="../output/surgery_iter",
        action_mask_scope="mid",
    )
    print(f"Saved to: {out}")
