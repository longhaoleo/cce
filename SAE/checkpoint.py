"""
SharedSAE checkpoint 读写。

这层的职责很克制：

- 把当前训练配置和模型状态打包落盘
- 在恢复时把 checkpoint 配置和参数一并取回

它不负责：
- 推断最新 checkpoint 路径
- 决定应该恢复到哪个训练阶段
- 做向后兼容的复杂迁移

这些策略应该放在更上层的训练入口或运行时入口里。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import TrainConfig


def save_checkpoint(
    *,
    output_root: str,
    stage: str,
    global_step: int,
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    norm_scale_by_block: Dict[str, float],
    extra: Optional[Dict] = None,
) -> Path:
    """保存训练 checkpoint。

    这里写出的 `config.json` 不是简单原样抄配置，而是额外补了一些
    “运行时重建模型必须知道”的结构性字段，例如：

    - `shared_sae`
    - `global_feature_space`
    - `block_adapter_type`
    - `time_pos_encoding`
    - `spatial_pos_encoding`

    这样后面的推理脚本只靠 checkpoint 目录就能尽量自洽地重建模型。
    """
    ckpt_dir = Path(output_root).expanduser().resolve() / "checkpoints" / f"{stage}_step_{int(global_step):07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = ckpt_dir / "config.json"
    state_path = ckpt_dir / "state_dict.pt"
    opt_path = ckpt_dir / "optimizer.pt"

    payload = cfg.to_dict()
    # 这些字段本质上是在把“当前实验约定”显式写进 checkpoint，
    # 避免后面运行时只能靠默认值猜结构。
    payload["shared_sae"] = True
    payload["global_feature_space"] = True
    payload["block_adapter_type"] = "lora"
    payload["time_pos_encoding"] = "sincos_1d"
    payload["spatial_pos_encoding"] = "sincos_2d"
    payload["loss_recon"] = "mse"
    payload["loss_auxk"] = True
    payload["loss_align"] = "mid_anchor_pooled_l2_sq"
    payload["time_space_interaction"] = False
    payload["use_block_in_adapter"] = bool(getattr(model, "use_block_in_adapter", payload.get("use_block_in_adapter", False)))
    payload["use_block_out_adapter"] = bool(
        getattr(model, "use_block_out_adapter", payload.get("use_block_out_adapter", False))
    )
    payload["norm_scale_by_block"] = {k: float(v) for k, v in norm_scale_by_block.items()}
    payload["stage"] = str(stage)
    payload["global_step"] = int(global_step)
    if extra:
        payload["extra"] = dict(extra)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    torch.save({"state_dict": model.state_dict()}, state_path)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict()}, opt_path)
    return ckpt_dir


def load_checkpoint(
    *,
    ckpt_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict:
    """加载 checkpoint 并恢复参数。

    返回值直接是 `config.json` 里的内容，而不是重新封装成 dataclass。
    这样运行时脚本可以自由读取其中的额外字段，不必被训练配置结构强约束。
    """
    path = Path(ckpt_dir).expanduser().resolve()
    cfg_path = path / "config.json"
    state_path = path / "state_dict.pt"
    opt_path = path / "optimizer.pt"

    if not cfg_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"checkpoint 文件缺失: {path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_state = torch.load(state_path, map_location=map_location)
    model.load_state_dict(model_state["state_dict"], strict=True)

    if optimizer is not None and opt_path.exists():
        opt_state = torch.load(opt_path, map_location=map_location)
        optimizer.load_state_dict(opt_state["optimizer"])
    return cfg
