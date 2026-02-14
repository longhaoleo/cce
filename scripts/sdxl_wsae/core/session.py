"""实验会话：封装模型读取、SAE 读取与轨迹采样。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from SAE.sae import SparseAutoencoder

from ..configs import ModelConfig, RunConfig, SAEConfig


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """将字符串精度映射为 torch.dtype。"""
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = (dtype_name or "fp16").lower()
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return mapping[key]


def _resolve_device_dtype(device: str, dtype: torch.dtype) -> Tuple[str, torch.dtype]:
    """根据硬件条件修正 device/dtype 组合。"""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU + float32。")
        return "cpu", torch.float32
    if device == "cpu" and dtype == torch.float16:
        return "cpu", torch.float32
    return device, dtype


def _import_hooked_sdxl_pipeline_cls():
    """导入仓库内置的 HookedStableDiffusionXLPipeline（SDLens 已在仓库中）。"""
    from SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline

    return HookedStableDiffusionXLPipeline


class SAECheckpointResolver:
    """按 block 名称解析 SAE 检查点路径。"""

    def __init__(self, root: str, *, prefer_k: int = 10, prefer_hidden: int = 5120):
        self.root = Path(os.path.expanduser(root)).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"SAE 根目录不存在: {self.root}")
        self.prefer_k = int(prefer_k)
        self.prefer_hidden = int(prefer_hidden)

    @staticmethod
    def _is_sae_dir(path: Path) -> bool:
        return path.is_dir() and (path / "config.json").exists() and (path / "state_dict.pth").exists()

    def _candidate_dirs(self, block: str) -> List[Path]:
        candidates: List[Path] = []
        roots = [self.root, self.root / "checkpoints"]
        for root in roots:
            if not root.exists():
                continue

            for p in (root / block / "final", root / block):
                if self._is_sae_dir(p):
                    candidates.append(p)

            for p in sorted(root.glob(f"{block}*")):
                if self._is_sae_dir(p):
                    candidates.append(p)
                if self._is_sae_dir(p / "final"):
                    candidates.append(p / "final")

        uniq: List[Path] = []
        seen = set()
        for p in candidates:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                uniq.append(p)
        return uniq

    def resolve(self, block: str) -> Path:
        cands = self._candidate_dirs(block)
        if not cands:
            raise FileNotFoundError(f"未找到 block={block} 的 SAE 检查点，搜索根目录: {self.root}")

        prefer_tag = f"_k{self.prefer_k}_hidden{self.prefer_hidden}_"
        chosen = None
        for path in cands:
            name = path.parent.name if path.name == "final" else path.name
            if prefer_tag in name:
                chosen = path
                break
        if chosen is None:
            chosen = cands[0]

        print(f"解析 SAE 检查点: block={block} -> {chosen}")
        return chosen


class SDXLExperimentSession:
    """
    SDXL 实验会话对象。

    设计目标：
    - 模型加载与实验逻辑解耦；
    - SAE 加载集中管理；
    - 采样入口统一，便于不同实验复用。
    """

    def __init__(self, model_cfg: ModelConfig, sae_cfg: SAEConfig):
        self.model_cfg = model_cfg
        self.sae_cfg = sae_cfg
        self.device, self.dtype = _resolve_device_dtype(
            model_cfg.device,
            _resolve_dtype(model_cfg.dtype_name),
        )
        self.HookedStableDiffusionXLPipeline = _import_hooked_sdxl_pipeline_cls()
        self.pipe = self._load_pipeline()
        self.resolver = SAECheckpointResolver(
            sae_cfg.sae_root,
            prefer_k=sae_cfg.prefer_k,
            prefer_hidden=sae_cfg.prefer_hidden,
        )
        self.saes: Dict[str, torch.nn.Module] = {}

    def _load_pipeline(self):
        """
        加载 SDXL pipeline，并尽量使用目标 dtype。

        采用两段回退策略：
        1) 先带 `torch_dtype` 加载；
        2) 若失败，再不带 dtype 参数重试。
        """
        model_id = os.path.expanduser(self.model_cfg.model_id)
        attempts = [{"torch_dtype": self.dtype}, {}]
        last_err: Optional[BaseException] = None
        for kwargs in attempts:
            try:
                print(f"加载 SDXL pipeline: {model_id} kwargs={list(kwargs.keys())}")
                pipe = self.HookedStableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)
                return pipe.to(self.device)
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    def load_saes(self, blocks: Optional[Sequence[str]] = None) -> Dict[str, torch.nn.Module]:
        """加载指定 block 的 SAE；若为 None 则加载 SAEConfig 中全部 block。"""
        target_blocks = list(blocks) if blocks is not None else list(self.sae_cfg.blocks)
        for block in target_blocks:
            if block in self.saes:
                continue
            ckpt = self.resolver.resolve(block)
            sae = SparseAutoencoder.load_from_disk(str(ckpt))
            sae = sae.to(self.device, dtype=self.dtype).eval()
            self.saes[block] = sae
            print(f"已加载 SAE: {block} -> {ckpt}")
        return self.saes

    def get_sae(self, block: str) -> torch.nn.Module:
        """获取单个 block 的 SAE，必要时自动加载。"""
        if block not in self.saes:
            self.load_saes([block])
        return self.saes[block]

    @staticmethod
    def scheduler_timesteps(pipe: Any) -> List[int]:
        """读取 scheduler 时间步序列。"""
        ts = getattr(getattr(pipe, "scheduler", None), "timesteps", None)
        if ts is None:
            return []
        if isinstance(ts, torch.Tensor):
            return [int(x) for x in ts.detach().cpu().tolist()]
        return [int(x) for x in ts]

    @staticmethod
    def _make_generator(seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(int(seed))

    @staticmethod
    def set_global_seed(seed: int) -> None:
        """设置全局随机种子，保证可复现实验。"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_with_cache(
        self,
        run_cfg: RunConfig,
        *,
        positions_to_cache: Sequence[str],
        save_input: bool = True,
        save_output: bool = True,
        output_type: str = "pil",
    ):
        """执行一次采样并缓存指定位置的输入/输出。"""
        self.set_global_seed(int(run_cfg.seed))
        generator = self._make_generator(run_cfg.seed)
        return self.pipe.run_with_cache(
            prompt=run_cfg.prompt,
            num_inference_steps=int(run_cfg.steps),
            guidance_scale=float(run_cfg.guidance_scale),
            generator=generator,
            positions_to_cache=list(positions_to_cache),
            save_input=save_input,
            save_output=save_output,
            output_type=output_type,
        )

    def run_with_hooks_and_cache(
        self,
        run_cfg: RunConfig,
        *,
        position_hook_dict: Dict[str, Any],
        positions_to_cache: Sequence[str],
        save_input: bool = True,
        save_output: bool = True,
        output_type: str = "pil",
    ):
        """执行带 hook 的采样，并缓存指定位置数据。"""
        self.set_global_seed(int(run_cfg.seed))
        generator = self._make_generator(run_cfg.seed)
        return self.pipe.run_with_hooks_and_cache(
            prompt=run_cfg.prompt,
            num_inference_steps=int(run_cfg.steps),
            guidance_scale=float(run_cfg.guidance_scale),
            generator=generator,
            output_type=output_type,
            position_hook_dict=position_hook_dict,
            positions_to_cache=list(positions_to_cache),
            save_input=save_input,
            save_output=save_output,
        )
