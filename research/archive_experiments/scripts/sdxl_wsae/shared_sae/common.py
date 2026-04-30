"""SharedSAE 正式测试脚本的公共能力。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline

from .defaults import DEFAULT_BLOCKS
from SAE import SharedSAE, build_coords_norm, load_checkpoint


DEFAULT_SHARED_BLOCKS = tuple(str(x) for x in DEFAULT_BLOCKS)
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MODEL_LOCAL_DIR = "/root/autodl-tmp/models/sd-xl-base-1.0-fp16-only"


@dataclass
class SharedCheckpointBundle:
    """SharedSAE checkpoint 的运行时句柄。"""

    ckpt_dir: Path
    config: Dict[str, Any]
    model: SharedSAE
    norm_scale_by_block: Dict[str, float]


def add_checkpoint_args(group: argparse._ArgumentGroup) -> None:
    """添加 SharedSAE checkpoint 解析参数。"""
    group.add_argument(
        "--output_root",
        type=str,
        default="",
        help="训练输出根目录；若不传 ckpt_dir，则自动从这里找最新 checkpoint。",
    )
    group.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="直接指定 SharedSAE checkpoint 目录（优先级高于 output_root）。",
    )


def add_model_args(group: argparse._ArgumentGroup) -> None:
    """添加 SDXL 模型加载参数。"""
    group.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Diffusers 模型标识。",
    )
    group.add_argument(
        "--model_local_dir",
        type=str,
        default=DEFAULT_MODEL_LOCAL_DIR,
        help="本地 SDXL 目录；存在时优先离线加载。",
    )
    group.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否只用本地缓存或本地目录。",
    )
    group.add_argument("--device", type=str, default="cuda", help="运行设备。")
    group.add_argument("--dtype", type=str, default="fp16", help="推理精度：fp16/bf16/fp32。")


def add_generation_override_args(
    group: argparse._ArgumentGroup,
    *,
    prompt_required: bool,
) -> None:
    """添加从 checkpoint 继承的采样参数。"""
    if prompt_required:
        group.add_argument("--prompt", type=str, required=True, help="待测试 prompt。")
    group.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="扩散步数；<0 时继承 checkpoint 配置。",
    )
    group.add_argument(
        "--guidance_scale",
        type=float,
        default=-1.0,
        help="CFG 强度；<0 时继承 checkpoint 配置。",
    )
    group.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="生成分辨率；<=0 时继承 checkpoint 配置。",
    )
    group.add_argument("--seed", type=int, default=42, help="采样种子。")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """将 dtype 名称解析为 torch.dtype。"""
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(dtype_name or "fp16").lower()
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return mapping[key]


def resolve_device_dtype(device: str, dtype: torch.dtype, *, log_prefix: str) -> Tuple[str, torch.dtype]:
    """在 CPU/无 CUDA 时做安全回退。"""
    if device == "cuda" and not torch.cuda.is_available():
        print(f"[{log_prefix}] CUDA 不可用，自动回退到 CPU + float32。")
        return "cpu", torch.float32
    if device == "cpu" and dtype == torch.float16:
        return "cpu", torch.float32
    return device, dtype


def load_hooked_pipeline(
    *,
    model_id: str,
    model_local_dir: str,
    local_files_only: bool,
    device: str,
    dtype: torch.dtype,
    log_prefix: str,
):
    """按训练脚本同样的优先级加载 Hooked SDXL。"""
    candidates: List[Tuple[str, Dict[str, object]]] = []
    local_dir = str(model_local_dir or "").strip()
    if local_dir:
        path = Path(local_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--model_local_dir 不存在: {path}")
        src = str(path)
        candidates.extend(
            [
                (src, {"torch_dtype": dtype, "variant": "fp16", "use_safetensors": True, "local_files_only": True}),
                (src, {"torch_dtype": dtype, "use_safetensors": True, "local_files_only": True}),
                (src, {"local_files_only": True}),
            ]
        )
    else:
        src = str(model_id)
        base_kwargs: List[Dict[str, object]] = [
            {"torch_dtype": dtype, "variant": "fp16", "use_safetensors": True},
            {"torch_dtype": dtype, "use_safetensors": True},
            {},
        ]
        if local_files_only:
            base_kwargs = [dict(kwargs, local_files_only=True) for kwargs in base_kwargs]
        candidates.extend((src, kwargs) for kwargs in base_kwargs)

    last_err: Exception | None = None
    for src, kwargs in candidates:
        try:
            print(f"[{log_prefix}] 尝试加载 SDXL: source={src} kwargs={kwargs}")
            pipe = HookedStableDiffusionXLPipeline.from_pretrained(src, **kwargs)
            return pipe.to(device)
        except Exception as exc:  # pragma: no cover - 真实运行分支
            last_err = exc
            print(f"[{log_prefix}] 当前加载方案失败，继续尝试下一种。err={exc}")
    raise RuntimeError(
        "加载 SDXL 失败。\n"
        "可尝试：\n"
        "1) 传 --model_local_dir 指向本地 SDXL 目录\n"
        "2) 或传 --local_files_only 并确保缓存已就绪\n"
        "3) 或检查网络/证书链"
    ) from last_err


def resolve_checkpoint_dir(*, ckpt_dir: str, output_root: str) -> Path:
    """解析 checkpoint 目录。"""
    if str(ckpt_dir or "").strip():
        path = Path(ckpt_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--ckpt_dir 不存在: {path}")
        return path

    output_root = str(output_root or "").strip()
    if not output_root:
        raise ValueError("必须至少提供 --ckpt_dir 或 --output_root 其一。")
    root = Path(output_root).expanduser().resolve()
    ckpt_root = root / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"未找到 checkpoints 目录: {ckpt_root}")

    def _step_of(path: Path) -> int:
        match = re.search(r"_step_(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    candidates = [
        path
        for path in ckpt_root.iterdir()
        if path.is_dir() and (path / "config.json").exists() and (path / "state_dict.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"未在 {ckpt_root} 下找到合法 checkpoint 目录。")
    candidates.sort(key=_step_of)
    return candidates[-1]


def build_shared_sae_from_cfg(cfg: Dict[str, Any]) -> SharedSAE:
    """根据 checkpoint 配置重建 SharedSAE。"""
    d_model = int(cfg.get("d_model", 1280))
    n_dirs = int(cfg.get("n_dirs", int(cfg.get("expansion_factor", 4)) * d_model))
    blocks = tuple(str(x) for x in cfg.get("blocks", list(DEFAULT_SHARED_BLOCKS)))
    return SharedSAE(
        blocks=blocks,
        d_model=d_model,
        n_dirs=n_dirs,
        top_k=int(cfg.get("top_k", 10)),
        auxk=int(cfg.get("auxk", 256)),
        dead_tokens_threshold=int(cfg.get("dead_tokens_threshold", 10_000_000)),
        use_block_in_adapter=bool(cfg.get("use_block_in_adapter", True)),
        use_block_out_adapter=bool(cfg.get("use_block_out_adapter", False)),
        block_in_rank=int(cfg.get("block_in_rank", 16)),
        block_in_alpha=int(cfg.get("block_in_alpha", 16)),
        block_out_rank=int(cfg.get("block_out_rank", 16)),
        block_out_alpha=int(cfg.get("block_out_alpha", 16)),
        use_time_branch=bool(cfg.get("use_time_branch", True)),
        time_branch_mode=str(cfg.get("time_branch_mode", "sincos_linear")),
        time_embed_dim=int(cfg.get("time_embed_dim", 32)),
        time_hidden_dim=int(cfg.get("time_hidden_dim", 128)),
        use_spatial_branch=bool(cfg.get("use_spatial_branch", True)),
        spatial_branch_mode=str(cfg.get("spatial_branch_mode", "sincos_linear")),
        spatial_embed_dim=int(cfg.get("spatial_embed_dim", 64)),
        spatial_hidden_dim=int(cfg.get("spatial_hidden_dim", 128)),
    )


def load_shared_checkpoint_bundle(
    *,
    ckpt_dir: Path | str,
    device: str,
    dtype: torch.dtype,
) -> SharedCheckpointBundle:
    """加载 SharedSAE checkpoint 与归一化元数据。"""
    path = Path(ckpt_dir).expanduser().resolve()
    cfg_path = path / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    model = build_shared_sae_from_cfg(ckpt_cfg)
    loaded_cfg = load_checkpoint(ckpt_dir=str(path), model=model, optimizer=None, map_location="cpu")
    model = model.to(device=device, dtype=dtype).eval()
    norm_scale_by_block = {
        str(key): float(value)
        for key, value in (loaded_cfg.get("norm_scale_by_block", {}) or {}).items()
    }
    return SharedCheckpointBundle(
        ckpt_dir=path,
        config=loaded_cfg,
        model=model,
        norm_scale_by_block=norm_scale_by_block,
    )


def resolve_blocks(*, requested_blocks: Sequence[str] | None, ckpt_cfg: Dict[str, Any]) -> List[str]:
    """优先使用命令行 block，否则继承 checkpoint block 列表。"""
    if requested_blocks:
        return [str(x) for x in requested_blocks]
    return [str(x) for x in ckpt_cfg.get("blocks", list(DEFAULT_SHARED_BLOCKS))]


def resolve_norm_scale_by_block(
    *,
    bundle: SharedCheckpointBundle,
    blocks: Sequence[str],
    log_prefix: str,
    warn_if_missing: bool,
) -> Dict[str, float]:
    """校验 block 合法性并补齐 norm scale。"""
    norm_scale_by_block = dict(bundle.norm_scale_by_block)
    for block in blocks:
        if block not in bundle.model.block_set:
            raise ValueError(f"block 不在 checkpoint 模型中: {block}")
        if block not in norm_scale_by_block:
            if warn_if_missing:
                print(f"[{log_prefix}] 警告：block={block} 未在 checkpoint 中保存 norm_scale，回退为 1.0")
            norm_scale_by_block[block] = 1.0
    return norm_scale_by_block


def resolve_generation_hparams(
    *,
    args: argparse.Namespace,
    ckpt_cfg: Dict[str, Any],
) -> Tuple[int, float, int]:
    """按“命令行优先，否则继承 checkpoint”解析采样参数。"""
    steps = int(args.steps) if int(args.steps) > 0 else int(ckpt_cfg.get("steps", 30))
    guidance_scale = float(args.guidance_scale) if float(args.guidance_scale) >= 0 else float(ckpt_cfg.get("guidance_scale", 8.0))
    resolution = int(args.resolution) if int(args.resolution) > 0 else int(ckpt_cfg.get("resolution", 512))
    return steps, guidance_scale, resolution


def scheduler_timesteps(pipe) -> List[int]:
    """读取 scheduler timesteps。"""
    ts = getattr(getattr(pipe, "scheduler", None), "timesteps", None)
    if ts is None:
        return []
    if isinstance(ts, torch.Tensor):
        return [int(x) for x in ts.detach().cpu().tolist()]
    return [int(x) for x in ts]


def make_generator(seed: int) -> torch.Generator:
    """构造稳定的 CPU generator。"""
    return torch.Generator(device="cpu").manual_seed(int(seed))


def coords_for_hw(
    *,
    hw: Tuple[int, int],
    n_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """根据 (H, W) 与 token 数还原 coords_norm。"""
    h, w = int(hw[0]), int(hw[1])
    per_map = max(1, h * w)
    batch_repeat = max(1, int(n_tokens) // per_map)
    coords = build_coords_norm(h, w, device=device, dtype=dtype)
    return coords.repeat(batch_repeat, 1)


def meta_hw(meta: Tuple[str, int, int, int]) -> Tuple[int, int, int]:
    """从 flatten meta 中恢复 (batch, h, w)。"""
    kind, a, b, c = meta
    if kind == "bchw":
        return int(a), int(b), int(c)
    if kind == "bnc":
        bsz, n = int(a), int(b)
        side = int(round(float(n) ** 0.5))
        if side > 0 and side * side == n:
            return bsz, side, side
        return bsz, 1, n
    raise ValueError(f"未知 meta: {meta}")


def coords_from_meta(
    meta: Tuple[str, int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """根据 hook flatten meta 还原 coords_norm。"""
    bsz, h, w = meta_hw(meta)
    coords = build_coords_norm(h, w, device=device, dtype=dtype)
    return coords.repeat(int(bsz), 1)
