"""
自定义 Attention Processor：实现“定位-切除-补偿”，并支持可选注意力捕获。
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from diffusers.models.attention_processor import AttnProcessor


@dataclass
class CaptureMapStore:
    """保存多次捕获的 2D attention map。"""
    maps: List[np.ndarray] = field(default_factory=list)

    def add(self, m: np.ndarray):
        """追加一张 2D attention map。"""
        self.maps.append(m)

    def mean(self) -> Optional[np.ndarray]:
        """返回所有 map 的平均结果。"""
        if not self.maps:
            return None
        return np.mean(np.stack(self.maps, axis=0), axis=0)


class CausalSurgeryAttnProcessor(AttnProcessor):
    """
    Cross-attention Processor：实现“定位-切除-补偿”逻辑。

    - 接收预先计算好的 surgery_mask（2D），并在每层动态缩放到当前分辨率。
    - 在 softmax 后、与 value 相乘之前：
      1) 把客体 token 在 mask 区域内的权重置零
      2) 将被切掉的能量加回到主体 token
      3) 可选做归一化，确保 attention 在 token 维度和为 1
    """

    def __init__(
        self,
        subject_token_indices: List[int],
        object_token_indices: List[int],
        surgery_mask: Optional[torch.Tensor] = None,  # shape (H, W) or (1,1,H,W)
        apply_to_cond_only: bool = True,
        renorm: bool = True,
        eps: float = 1e-6,
        capture_token_indices: Optional[List[int]] = None,
        capture_store: Optional[CaptureMapStore] = None,
        capture_target_hw: Optional[Tuple[int, int]] = None,
        capture_only_cond: bool = True,
    ):
        """初始化手术 processor 与可选捕获配置。"""
        super().__init__()
        self.subject_token_indices = list(subject_token_indices)
        self.object_token_indices = list(object_token_indices)
        self.apply_to_cond_only = apply_to_cond_only
        self.renorm = renorm
        self.eps = eps
        self.capture_token_indices = list(capture_token_indices) if capture_token_indices else []
        self.capture_store = capture_store
        self.capture_target_hw = capture_target_hw
        self.capture_only_cond = capture_only_cond
        self._mask = None
        if surgery_mask is not None:
            self.set_surgery_mask(surgery_mask)

    def set_surgery_mask(self, surgery_mask: torch.Tensor):
        """
        接收以下形状：
          - (H, W)
          - (1, 1, H, W)
        存储为 float，不做硬截断。
        """
        if surgery_mask.ndim == 2:
            surgery_mask = surgery_mask.unsqueeze(0).unsqueeze(0)
        if surgery_mask.ndim != 4:
            raise ValueError("surgery_mask must be (H,W) or (1,1,H,W).")
        self._mask = surgery_mask.float()

    def _resize_mask(self, q_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        将 mask resize 到当前 q_len（H*W），再展平成 (1, 1, Q, 1)。
        若 q_len 不是平方数则返回 None。
        """
        if self._mask is None:
            return None
        side = int(math.sqrt(q_len))
        if side * side != q_len:
            return None
        mask = self._mask.to(device=device, dtype=dtype)
        if mask.shape[-2:] != (side, side):
            mask = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False)
        mask = mask.reshape(1, 1, q_len, 1)
        return mask

    def _apply_surgery(self, attn_4d: torch.Tensor, mask_q: torch.Tensor) -> torch.Tensor:
        """
        attn_4d: [B, H, Q, K]
        mask_q:  [1, 1, Q, 1]
        """
        if not self.object_token_indices or not self.subject_token_indices:
            return attn_4d

        obj_idx = self.object_token_indices
        subj_idx = self.subject_token_indices

        smoke_weights = attn_4d[:, :, :, obj_idx]  # [B, H, Q, O]
        mask_q = mask_q.to(dtype=attn_4d.dtype)
        smoke_new = smoke_weights * (1.0 - mask_q)
        energy_lost = smoke_weights * mask_q  # [B, H, Q, O]

        attn_4d[:, :, :, obj_idx] = smoke_new

        # 将被切掉的能量分配给主体 tokens（保持 rank=4 以避免广播错位）
        lost_sum = energy_lost.sum(dim=-1, keepdim=True)  # [B, H, Q, 1]
        scale = lost_sum.new_tensor(1.0 / max(len(subj_idx), 1))
        distribute = lost_sum * scale                    # [B, H, Q, 1]
        distribute = distribute.to(dtype=attn_4d.dtype)
        attn_4d[:, :, :, subj_idx] = attn_4d[:, :, :, subj_idx].add(distribute)

        # 可选：重新归一化，避免数值累积带来的偏移
        if self.renorm:
            denom = attn_4d.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            attn_4d = attn_4d / denom
        return attn_4d

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """执行注意力手术，并可选捕获指定 token 的注意力图。"""
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

        if is_cross and self._mask is not None:
            heads = attn.heads
            bh, q_len, k_len = attention_probs.shape
            bsz = bh // heads
            attn_4d = attention_probs.view(bsz, heads, q_len, k_len)

            mask_q = self._resize_mask(q_len, attention_probs.device, attention_probs.dtype)
            if mask_q is not None:
                # CFG 情况下仅对 cond 分支生效
                if self.apply_to_cond_only and (bsz % 2 == 0) and (bsz >= 2):
                    uncond = attn_4d[: bsz // 2]
                    cond = attn_4d[bsz // 2 :]
                    cond = self._apply_surgery(cond, mask_q)
                    attn_4d = torch.cat([uncond, cond], dim=0)
                else:
                    attn_4d = self._apply_surgery(attn_4d, mask_q)

                # 可选：捕获指定 token 的注意力图（在 surgery 之后）
                if self.capture_store is not None and self.capture_token_indices and self.capture_target_hw is not None:
                    side = int(math.sqrt(q_len))
                    if side * side == q_len and (side, side) == self.capture_target_hw:
                        if self.capture_only_cond and (bsz % 2 == 0) and (bsz >= 2):
                            cap_attn = attn_4d[bsz // 2 :]
                        else:
                            cap_attn = attn_4d
                        # 平均 batch 和 heads -> [Q, K]
                        cap_qk = cap_attn.mean(dim=0).mean(dim=0)
                        vec = cap_qk[:, self.capture_token_indices].sum(dim=1)
                        m = vec.view(side, side).detach().float().cpu().numpy()
                        self.capture_store.add(m)

                attention_probs = attn_4d.view(bh, q_len, k_len)

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


def attach_surgery_processors(
    pipe,
    subject_token_indices: List[int],
    object_token_indices: List[int],
    surgery_mask: torch.Tensor,
    apply_to_blocks: Tuple[str, ...] = ("mid", "up"),
    apply_to_cond_only: bool = True,
    renorm: bool = True,
    capture_token_indices: Optional[List[int]] = None,
    capture_store: Optional[CaptureMapStore] = None,
    capture_target_hw: Optional[Tuple[int, int]] = None,
    capture_only_cond: bool = True,
):
    """
    将 CausalSurgeryAttnProcessor 挂载到指定 block 的 cross-attn (attn2) 上。
    apply_to_blocks: ("mid", "up", "down")
    """
    orig_procs = dict(pipe.unet.attn_processors)
    patched = {}
    for name, proc in orig_procs.items():
        is_attn2 = ".attn2." in name
        is_mid = name.startswith("mid_block")
        is_up = name.startswith("up_blocks")
        is_down = name.startswith("down_blocks")

        use_mid = "mid" in apply_to_blocks and is_mid
        use_up = "up" in apply_to_blocks and is_up
        use_down = "down" in apply_to_blocks and is_down

        if is_attn2 and (use_mid or use_up or use_down):
            patched[name] = CausalSurgeryAttnProcessor(
                subject_token_indices=subject_token_indices,
                object_token_indices=object_token_indices,
                surgery_mask=surgery_mask,
                apply_to_cond_only=apply_to_cond_only,
                renorm=renorm,
                capture_token_indices=capture_token_indices,
                capture_store=capture_store,
                capture_target_hw=capture_target_hw,
                capture_only_cond=capture_only_cond,
            )
        else:
            patched[name] = proc
    pipe.unet.set_attn_processor(patched)
    return orig_procs
