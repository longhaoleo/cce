"""
实验共享：一次采样 + 缓存 + delta 提取。

为什么要单独拆出来：
- exp51（Top-K 叠加热图）和 exp52（瀑布图 Money Plot）都需要同样的前置步骤：
  1) 跑一次推理轨迹，并缓存指定 hookpoint 的 input/output；
  2) 对每个 step 计算 delta = h_out - h_in；
  3) 把 delta 整理成 SAE 可编码的 token 形式。
- 这部分如果复制到每个实验里，会导致维护困难（参数/缓存结构一变就要改很多处）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..configs import RunConfig, SAEConfig
from ..core.session import SDXLExperimentSession
from ..utils import ensure_dir, extract_first_image


@dataclass
class StepDelta:
    """单个去噪 step 的 delta 表示。

    - step_idx: 该 delta 来自第几个去噪 step（0..N-1）
    - timestep: scheduler 的实际时间步编号（通常从大到小，如 1000 -> 0）
    - x: 拉平成 token 的 delta 张量，形状一般是 [tokens, d_model]
    - hw: 如果能恢复到 2D（如 U-Net 的 feature map），这里记录 (H, W) 以便画空间热图
    """

    step_idx: int
    timestep: int
    x: torch.Tensor
    hw: Tuple[int, int]


class DeltaExtractor:
    """从缓存中提取 delta，并整理成 SAE 可编码的 token 形式。"""

    @staticmethod
    def _select_conditional(x: torch.Tensor) -> torch.Tensor:
        """选取条件批次（跳过 unconditional），避免 CFG 把 batch 维度翻倍后混入统计。"""
        if x.dim() == 0:
            return x
        b = int(x.shape[0])
        if b == 2:
            return x[1:2]
        if b > 2 and b % 2 == 0:
            return x[b // 2 :]
        return x

    @staticmethod
    def _step_to_tokens(delta_step: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将单步 delta 转换为 SAE 可编码的 token 形式。"""
        if delta_step.dim() == 4:
            b, c, h, w = map(int, delta_step.shape)
            return delta_step.permute(0, 2, 3, 1).reshape(b * h * w, c), (h, w)
        if delta_step.dim() == 3:
            b, n, c = map(int, delta_step.shape)
            side = int(math.isqrt(n))
            if side * side == n:
                return delta_step.reshape(b * n, c), (side, side)
            return delta_step.reshape(b * n, c), (1, n)
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
            ts = int(timesteps[step_idx]) if step_idx < len(timesteps) else -1
            out.append(StepDelta(step_idx=step_idx, timestep=ts, x=x, hw=hw))
        return out


def prepare_deltas_for_blocks(
    *,
    model_cfg,
    sae_cfg: SAEConfig,
    run_cfg: RunConfig,
    output_dir: str,
    blocks: List[str],
) -> Tuple[SDXLExperimentSession, object, List[int], Dict[str, List[StepDelta]]]:
    """运行一次推理并准备后续可视化所需的中间量。

    返回：
    - session: SDXLExperimentSession（内含 pipe 与已加载的 saes）
    - base_image: 最终生成图（PIL.Image 或 None，类型用 object 避免这里引 PIL）
    - timesteps: scheduler 的 timestep 列表（长度通常为 steps）
    - deltas_by_block: Dict[block, List[StepDelta]]（每个 block 的每步 delta tokens）
    """
    ensure_dir(output_dir)
    session = SDXLExperimentSession(model_cfg, sae_cfg)
    session.load_saes(blocks)

    output, cache = session.run_with_cache(
        run_cfg,
        positions_to_cache=blocks,
        save_input=True,
        save_output=True,
        output_type="pil",
    )

    base_image = extract_first_image(output)
    if base_image is not None:
        base_image.save(os.path.join(output_dir, "generated_image.png"))

    timesteps = session.scheduler_timesteps(session.pipe)
    extractor = DeltaExtractor()
    deltas_by_block: Dict[str, List[StepDelta]] = {}
    for block in blocks:
        if block not in cache.get("input", {}) or block not in cache.get("output", {}):
            print(f"警告: block 未被缓存，已跳过: {block}")
            continue
        deltas_by_block[block] = extractor.extract(block=block, cache=cache, timesteps=timesteps)

    return session, base_image, timesteps, deltas_by_block
