"""
SharedSAE block 级输入归一化系数估计。

不同 block 的 token/update 范数经常差很多。
如果直接把这些 block 混在一起训练同一个 SharedSAE，模型会优先适配数值更大的 block，
从而削弱“共享字典”的意义。

这里的做法是：
1. 用 calibration prompt 估计每个 block 的平均 token 范数；
2. 把它们缩放到一个共同目标尺度附近；
3. 后续训练/推理都沿用这组 `block_scale`。
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from tqdm.auto import tqdm

from train.prompt_data import PromptRecord
from train.sampler import SDXLGroupSampler


def estimate_block_scales(
    *,
    sampler: SDXLGroupSampler,
    calibration_records: Sequence[PromptRecord],
    blocks: Sequence[str],
    d_model: int,
) -> Dict[str, float]:
    """估计每个 block 的归一化缩放系数 `s_b`。

    输出的目标是让每个 block 的平均 token 范数都落到 `sqrt(d_model)` 附近。
    这是一个常见且数值上比较自然的高维参考尺度，能让：

    - 不同 block 进入 Shared 编码器前量级更一致；
    - align / sparse / aux 分支不被某个超大 block 主导；
    - adapter 和 time/spatial 支路的训练更稳定。
    """
    sum_norm: Dict[str, float] = {b: 0.0 for b in blocks}
    count: Dict[str, int] = {b: 0 for b in blocks}

    for rec in tqdm(calibration_records, total=len(calibration_records), desc="calibration prompts", unit="prompt"):
        groups = sampler.sample_prompt_groups(
            prompt_id=int(rec.prompt_id),
            prompt=str(rec.prompt),
            seed=int(rec.seed),
            blocks=blocks,
        )
        for g in groups:
            for b in blocks:
                x = g.block_tokens[b]
                # 对每个 token 的最后一维做 L2 范数，再在 calibration 集上累计平均。
                norms = torch.norm(x, p=2, dim=-1)
                sum_norm[b] += float(norms.sum().item())
                count[b] += int(norms.numel())

    target = float(d_model) ** 0.5
    scales: Dict[str, float] = {}
    for b in blocks:
        if count[b] <= 0:
            raise RuntimeError(f"归一化统计失败，block={b} 无有效 token")
        mu_b = sum_norm[b] / float(count[b])
        scales[b] = target / max(mu_b, 1e-12)
    return scales


def apply_block_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """对输入 token 应用 block 级缩放。

    单独封装成函数，方便代码里显式表达“这里是在应用 calibration scale”，
    而不是把一个普通乘法散在各处。
    """
    return x * float(scale)
