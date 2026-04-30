"""SharedSAE 轨迹缓存的 delta 提取工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class StepDelta:
    """单个去噪 step 的 delta 表示。"""

    step_idx: int
    timestep: int
    x: torch.Tensor
    hw: Tuple[int, int]


class DeltaExtractor:
    """从缓存中提取 delta，并整理成 SAE 可编码的 token 形式。"""

    @staticmethod
    def _select_conditional(x: torch.Tensor) -> torch.Tensor:
        """选取条件批次，跳过 CFG 的 unconditional 部分。"""
        if x.dim() == 0:
            return x
        batch = int(x.shape[0])
        if batch == 2:
            return x[1:2]
        if batch > 2 and batch % 2 == 0:
            return x[batch // 2 :]
        return x

    @staticmethod
    def _step_to_tokens(delta_step: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将单步 delta 转成 SAE 可编码的 token 形式。"""
        if delta_step.dim() == 4:
            batch, channels, height, width = map(int, delta_step.shape)
            return delta_step.permute(0, 2, 3, 1).reshape(batch * height * width, channels), (height, width)
        if delta_step.dim() == 3:
            batch, n_tokens, channels = map(int, delta_step.shape)
            side = int(math.isqrt(n_tokens))
            if side * side == n_tokens:
                return delta_step.reshape(batch * n_tokens, channels), (side, side)
            return delta_step.reshape(batch * n_tokens, channels), (1, n_tokens)
        raise ValueError(f"不支持的 delta 形状: {tuple(delta_step.shape)}")

    def extract(
        self,
        *,
        block: str,
        cache: Dict[str, Dict[str, torch.Tensor]],
        timesteps: List[int],
    ) -> List[StepDelta]:
        """提取指定 block 的所有去噪步 delta 列表。"""
        h_in = cache["input"].get(block, None)
        h_out = cache["output"].get(block, None)
        if h_in is None or h_out is None:
            raise KeyError(f"缓存缺少 block={block} 的输入或输出。")
        if h_in.shape != h_out.shape:
            raise ValueError(f"输入输出形状不一致: {h_in.shape} vs {h_out.shape}")

        h_in = self._select_conditional(h_in)
        h_out = self._select_conditional(h_out)
        out: List[StepDelta] = []
        for step_idx in range(int(h_in.shape[1])):
            delta = h_out[:, step_idx] - h_in[:, step_idx]
            x, hw = self._step_to_tokens(delta)
            timestep = int(timesteps[step_idx]) if step_idx < len(timesteps) else -1
            out.append(StepDelta(step_idx=step_idx, timestep=timestep, x=x, hw=hw))
        return out

