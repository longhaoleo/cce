"""
SDXL 激活采样与 group 构建。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from diffusers import StableDiffusionXLPipeline

from SAE import build_coords_norm, check_expected_hw


@dataclass
class GroupSample:
    """单个 group 样本。

    输入：
    - prompt_id: prompt 全局编号。
    - step_idx: 采样的去噪步索引。
    - timestep: scheduler 的真实时间步。
    - coords_norm: Tensor[256,2]，共享坐标。
    - block_tokens: Dict[block, Tensor[256,d_model]]，每层 token 输入。
    - hw: 特征图尺寸 `(H,W)`。

    输出：
    - GroupSample，可直接喂给训练器。
    """

    prompt_id: int
    step_idx: int
    timestep: int
    coords_norm: torch.Tensor
    block_tokens: Dict[str, torch.Tensor]
    hw: Tuple[int, int]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """解析 dtype 字符串。

    输入：
    - dtype_name: `fp16/bf16/fp32` 等字符串。

    输出：
    - torch.dtype：对应张量类型。
    """
    key = str(dtype_name).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return mapping[key]


def _select_conditional_branch(x: torch.Tensor) -> torch.Tensor:
    """从 CFG batch 中选取 conditional 分支。

    输入：
    - x: Tensor[B,...]，可能包含 uncond/cond 拼接。

    输出：
    - Tensor：仅保留 conditional 分支。
    """
    if x.dim() == 0:
        return x
    b = int(x.shape[0])
    if b == 2:
        return x[1:2]
    if b > 2 and (b % 2 == 0):
        return x[b // 2 :]
    return x


def _delta_step_to_tokens(delta_step: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将单步 delta 转换为 token 形式。

    输入：
    - delta_step: Tensor，常见形状 `[B,C,H,W]` 或 `[B,N,C]`。

    输出：
    - tokens: Tensor[B*H*W,C] 或 Tensor[B*N,C]
    - hw: `(H,W)`，若无法还原平方网格则返回 `(1,N)`。
    """
    if delta_step.dim() == 4:
        b, c, h, w = map(int, delta_step.shape)
        tokens = delta_step.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return tokens, (h, w)
    if delta_step.dim() == 3:
        b, n, c = map(int, delta_step.shape)
        side = int(math.isqrt(n))
        if side * side == n:
            return delta_step.reshape(b * n, c), (side, side)
        return delta_step.reshape(b * n, c), (1, n)
    raise ValueError(f"不支持的 delta_step 形状: {tuple(delta_step.shape)}")


def _select_step_indices(total_steps: int, num_buckets: int, rng: random.Random) -> List[int]:
    """按时间桶从总步数中选择 step 索引。

    输入：
    - total_steps: 当前轨迹总步数。
    - num_buckets: 桶数量。
    - rng: 随机数生成器。

    输出：
    - List[int]：每个时间桶随机 1 个 step 索引。
    """
    if total_steps <= 0:
        return []
    buckets = max(1, int(num_buckets))
    out: List[int] = []
    for i in range(buckets):
        left = (i * total_steps) // buckets
        right = ((i + 1) * total_steps) // buckets
        if right <= left:
            right = min(total_steps, left + 1)
        cand = list(range(left, right))
        if not cand:
            continue
        out.append(int(rng.choice(cand)))
    return sorted(set(out))


def _locate_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """根据路径字符串定位子模块。

    输入：
    - root: 根模块（通常是 `pipe.unet` 或 pipeline）。
    - path: 如 `unet.mid_block.attentions.0`。

    输出：
    - torch.nn.Module：目标模块。
    """
    current = root
    for part in str(path).split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
            continue
        current = getattr(current, part)
    return current


class SDXLGroupSampler:
    """从 SDXL 轨迹中提取训练 group 的采样器。"""

    def __init__(
        self,
        *,
        model_id: str,
        model_local_dir: str,
        local_files_only: bool,
        device: str,
        dtype_name: str,
        steps: int,
        guidance_scale: float,
        num_step_buckets: int,
        resolution: int,
        expected_h: int,
        expected_w: int,
    ):
        """初始化采样器并加载 SDXL 管线。

        输入：
        - model_id/model_local_dir/local_files_only: 模型加载参数。
        - steps/guidance_scale: 采样控制参数。
        - num_step_buckets: 每条 prompt 保留多少个 step。
        - resolution: 生成图像分辨率。
        - expected_h/expected_w: 期望特征图尺寸；传 0 表示自动推断。

        输出：
        - SDXLGroupSampler 实例。
        """
        self.model_id = str(model_id)
        self.model_local_dir = str(model_local_dir or "").strip()
        self.local_files_only = bool(local_files_only)
        self.device = str(device)
        self.dtype = _resolve_dtype(dtype_name)
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.num_step_buckets = int(num_step_buckets)
        self.resolution = int(resolution)
        self.expected_h = int(expected_h)
        self.expected_w = int(expected_w)

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.dtype = torch.float32
        if self.device == "cpu" and self.dtype == torch.float16:
            self.dtype = torch.float32

        self.pipe = self._load_pipeline()
        self.pipe = self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def _load_pipeline(self) -> StableDiffusionXLPipeline:
        """按优先级加载 SDXL pipeline。

        输入：
        - 无（使用初始化参数）。

        输出：
        - StableDiffusionXLPipeline：可用的 pipeline 实例。
        """
        def _candidate_variants() -> List[str | None]:
            """根据当前 dtype 给出候选权重变体名。"""
            variants: List[str | None] = [None]
            if self.dtype == torch.float16:
                variants.append("fp16")
            if self.dtype == torch.bfloat16:
                variants.append("bf16")
            return variants

        # 优先使用显式本地目录（完全离线）。
        if self.model_local_dir:
            local_path = Path(self.model_local_dir).expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(f"--model_local_dir 不存在: {local_path}")
            last_err: Exception | None = None
            for variant in _candidate_variants():
                try:
                    kwargs = {
                        "torch_dtype": self.dtype,
                        "local_files_only": True,
                    }
                    if variant is not None:
                        kwargs["variant"] = variant
                    return StableDiffusionXLPipeline.from_pretrained(str(local_path), **kwargs)
                except Exception as e:
                    last_err = e
            raise RuntimeError(
                "本地模型目录加载失败。请确认目录下包含完整 diffusers 权重文件；"
                "若是 fp16-only/bf16-only 目录，当前代码也会自动尝试对应 variant。"
            ) from last_err

        # 若仅允许本地文件，则直接走离线缓存。
        if self.local_files_only:
            last_err: Exception | None = None
            for variant in _candidate_variants():
                try:
                    kwargs = {
                        "torch_dtype": self.dtype,
                        "local_files_only": True,
                    }
                    if variant is not None:
                        kwargs["variant"] = variant
                    return StableDiffusionXLPipeline.from_pretrained(self.model_id, **kwargs)
                except Exception as e:
                    last_err = e
            raise RuntimeError(
                "local_files_only=True 但本地缓存未命中。\n"
                "可选方案：\n"
                "1) 传入 --model_local_dir 指向本地 SDXL 目录；\n"
                "2) 先在可联网环境下载并同步缓存；\n"
                "3) 临时放开网络证书后重试在线下载。"
            ) from last_err

        # 默认尝试在线加载；若在线失败，再尝试本地缓存兜底。
        online_err: Exception | None = None
        for variant in _candidate_variants():
            try:
                kwargs = {
                    "torch_dtype": self.dtype,
                }
                if variant is not None:
                    kwargs["variant"] = variant
                return StableDiffusionXLPipeline.from_pretrained(self.model_id, **kwargs)
            except Exception as e:
                online_err = e

        local_err: Exception | None = None
        for variant in _candidate_variants():
            try:
                kwargs = {
                    "torch_dtype": self.dtype,
                    "local_files_only": True,
                }
                if variant is not None:
                    kwargs["variant"] = variant
                return StableDiffusionXLPipeline.from_pretrained(self.model_id, **kwargs)
            except Exception as e:
                local_err = e

        cause = online_err if online_err is not None else local_err
        raise RuntimeError(
            "模型加载失败：在线下载失败，且本地缓存也不存在。\n"
            f"model_id={self.model_id}\n"
            "建议：\n"
            "1) 使用 --model_local_dir /path/to/sdxl-base\n"
            "2) 或传 --local_files_only 并确保本地缓存已就绪\n"
            "3) 检查当前机器的 SSL 证书链/代理配置"
        ) from cause

    def _run_with_cache(self, prompt: str, seed: int, blocks: Sequence[str]) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]], List[int]]:
        """运行一次采样并缓存指定 block 的输入输出序列。

        输入：
        - prompt: 文本提示词。
        - seed: 随机种子。
        - blocks: 需要缓存的 block 名列表。

        输出：
        - cache_in: Dict[block, List[tensor]]，每步输入。
        - cache_out: Dict[block, List[tensor]]，每步输出。
        - timesteps: scheduler 时间步列表。
        """
        cache_in: Dict[str, List[torch.Tensor]] = {b: [] for b in blocks}
        cache_out: Dict[str, List[torch.Tensor]] = {b: [] for b in blocks}
        hooks = []

        def _build_hook(block_name: str):
            def _hook(module, inputs, output):
                x_in = inputs[0] if isinstance(inputs, tuple) else inputs
                x_out = output[0] if isinstance(output, tuple) else output
                if not isinstance(x_in, torch.Tensor) or not isinstance(x_out, torch.Tensor):
                    return
                cache_in[block_name].append(x_in.detach().cpu())
                cache_out[block_name].append(x_out.detach().cpu())

            return _hook

        for b in blocks:
            m = _locate_module(self.pipe, b)
            hooks.append(m.register_forward_hook(_build_hook(b)))

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        try:
            _ = self.pipe(
                prompt=str(prompt),
                num_inference_steps=int(self.steps),
                guidance_scale=float(self.guidance_scale),
                generator=gen,
                output_type="latent",
                height=int(self.resolution),
                width=int(self.resolution),
            )
        finally:
            for h in hooks:
                h.remove()

        ts = self.pipe.scheduler.timesteps
        if isinstance(ts, torch.Tensor):
            timesteps = [int(x) for x in ts.detach().cpu().tolist()]
        else:
            timesteps = [int(x) for x in list(ts)]
        return cache_in, cache_out, timesteps

    def sample_prompt_groups(
        self,
        *,
        prompt_id: int,
        prompt: str,
        seed: int,
        blocks: Sequence[str],
    ) -> List[GroupSample]:
        """对单条 prompt 采样并构造多个 group。

        输入：
        - prompt_id: prompt 编号。
        - prompt: 文本提示。
        - seed: 随机种子。
        - blocks: 当前阶段使用的 block 列表。

        输出：
        - List[GroupSample]：按时间桶抽样得到的 group 列表。
        """
        cache_in, cache_out, timesteps = self._run_with_cache(prompt=prompt, seed=seed, blocks=blocks)
        if not blocks:
            return []

        # 以第一层步数作为基准。
        first_block = str(blocks[0])
        total_steps = min(len(cache_in[first_block]), len(cache_out[first_block]), len(timesteps))
        rng = random.Random(int(seed) + int(prompt_id))
        selected = _select_step_indices(total_steps=total_steps, num_buckets=self.num_step_buckets, rng=rng)
        groups: List[GroupSample] = []

        for step_idx in selected:
            block_tokens: Dict[str, torch.Tensor] = {}
            hw_ref: Tuple[int, int] | None = None
            coords_ref: torch.Tensor | None = None

            for b in blocks:
                x_in = _select_conditional_branch(cache_in[b][step_idx])
                x_out = _select_conditional_branch(cache_out[b][step_idx])
                delta = x_out - x_in
                tokens, hw = _delta_step_to_tokens(delta)
                if hw_ref is None:
                    if self.expected_h > 0 and self.expected_w > 0:
                        check_expected_hw(hw, self.expected_h, self.expected_w)
                    hw_ref = hw
                    coords_ref = build_coords_norm(
                        hw[0],
                        hw[1],
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                elif hw != hw_ref:
                    raise ValueError(
                        f"同一 group 内 block 特征图尺寸不一致: block={b} got={hw} ref={hw_ref}"
                    )
                block_tokens[b] = tokens.to(dtype=torch.float32)

            if hw_ref is None or coords_ref is None:
                continue
            groups.append(
                GroupSample(
                    prompt_id=int(prompt_id),
                    step_idx=int(step_idx),
                    timestep=int(timesteps[step_idx]),
                    coords_norm=coords_ref,
                    block_tokens=block_tokens,
                    hw=hw_ref,
                )
            )
        return groups
