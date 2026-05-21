"""SharedSAE 版概念擦除正式入口。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .features.hook_ops import (
    InterventionSpec,
    _conditional_slice,
    _extract_input_tensor,
    _extract_tensor,
    _flatten_spatial,
    _in_time_window,
    _maybe_apply_spatial_norm_weight,
    _pack_tensor,
    _resolve_feature_list,
    _unflatten_spatial,
)
from .features.intervention import (
    _build_block_scale_map,
    _interpolate_coeff_by_step,
    _load_coeff_by_step_from_exp53_csv,
    _save_hook_debug_csv,
)
from .features.scoring import _load_blacklist_ids
from .io_utils import block_short_name, ensure_dir, extract_first_image, safe_name
from .sae_layout import maybe_use_sae_layout

from .pipeline import (
    add_checkpoint_args,
    add_generation_override_args,
    add_model_args,
    coords_from_meta,
    load_hooked_pipeline,
    load_shared_checkpoint_bundle,
    make_generator,
    resolve_blocks,
    resolve_checkpoint_dir,
    resolve_device_dtype,
    resolve_dtype,
    resolve_generation_hparams,
    resolve_norm_scale_by_block,
)

from SAE import SharedSAE, _topk_keep


LOG_PREFIX = "shared-erase"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERVENTION_BLOCKS = [
    "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    "unet.up_blocks.0.attentions.0",
    "unet.up_blocks.0.attentions.1",
]


@dataclass(frozen=True)
class TimeInterventionConfig:
    """时间方向上的干预控制。"""

    use_stat_weight: bool = True
    stat_weight_scale: float = 1.0
    use_learned_weight: bool = False
    learned_weight_mode: str = "relative_window"
    learned_weight_scale: float = 1.0
    learned_weight_temperature: float = 1.0
    learned_weight_transform: str = "neutral_sigmoid"
    learned_weight_target_mean: float = 0.001
    learned_weight_smooth_radius: int = 2
    fuse_mode: str = "stat_only"
    t_start: int = 900
    t_end: int = 100
    step_start: Optional[int] = None
    step_end: Optional[int] = None


@dataclass(frozen=True)
class SpatialInterventionConfig:
    """空间方向上的干预控制。"""

    use_norm_weight: bool = True


@dataclass(frozen=True)
class SharedInterventionConfig:
    """SharedSAE 擦除脚本的结构化干预配置。"""

    mode: str = "ablation"
    scale: float = 50.0
    inject_scale: float = 50.0
    feature_top_k: int = 5
    projection_ridge: float = 1e-4
    use_out_adapter_for_decode: bool = False
    apply_only_conditional: bool = True
    max_delta_over_x: float = 0.0
    time: TimeInterventionConfig = field(default_factory=TimeInterventionConfig)
    spatial: SpatialInterventionConfig = field(default_factory=SpatialInterventionConfig)


@dataclass
class FeatureBundle:
    """单个 block 上待干预的特征集合。"""

    concept_name: str
    feature_ids: List[int]
    feature_scales: List[float]
    coeff_by_step: Dict[int, torch.Tensor]

    def to_manifest(self) -> Dict[str, object]:
        """导出 manifest 所需的轻量结构。"""
        return {
            "concept_name": str(self.concept_name),
            "feature_ids": list(self.feature_ids),
            "feature_scales": list(self.feature_scales),
        }


def build_parser() -> argparse.ArgumentParser:
    """构建 SharedSAE 概念擦除参数。"""
    parser = argparse.ArgumentParser(description="Prompt-level concept erase for SharedSAE checkpoints")
    g_io = parser.add_argument_group("输入输出")
    g_ckpt = parser.add_argument_group("SharedSAE checkpoint")
    g_model = parser.add_argument_group("SDXL")
    g_run = parser.add_argument_group("采样")
    g_int = parser.add_argument_group("概念干预")

    g_io.add_argument(
        "--output_dir",
        type=str,
        default="image_output/shared_concept_erase",
        help="图片与诊断文件输出目录。",
    )
    g_io.add_argument(
        "--case_number",
        type=int,
        default=0,
        help="保存到 eval_original/eval_erased 时使用的 case 编号。",
    )
    g_io.add_argument(
        "--sample_name",
        type=str,
        default="",
        help="评估图片文件名标签；默认用 targetconcept。",
    )
    g_io.add_argument(
        "--concept_root",
        type=str,
        default="concept_dict",
        help="SharedSAE 概念统计根目录；默认使用 concept_dict，并兼容旧 out_concept_dict_* 目录。",
    )
    g_io.add_argument(
        "--sae_root",
        type=str,
        default="",
        help="统一 SAE 产物根目录；传入后自动映射 concept-dig / blacklist 等子目录。",
    )

    add_checkpoint_args(g_ckpt)
    add_model_args(g_model)
    add_generation_override_args(g_run, prompt_required=True)
    g_run.add_argument("--no_baseline", action="store_true", help="只生成干预图，不生成 baseline。")

    add_intervention_args(g_int, include_targetconcept=True)
    return parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    return build_parser().parse_args()


def _step_or_none(value: int) -> Optional[int]:
    """将负值 step 边界转换为 None。"""
    return None if int(value) < 0 else int(value)


def add_intervention_args(
    group: argparse._ArgumentGroup,
    *,
    include_targetconcept: bool,
    default_blocks: Optional[List[str]] = None,
) -> None:
    """向 CLI 分组注入统一的 Shared 擦除参数。"""
    if include_targetconcept:
        group.add_argument("--targetconcept", type=str, required=True, help="概念名，对应 concept_root/<block>/<concept>/。")
    group.add_argument(
        "--injectconcept",
        type=str,
        default="",
        help="替换/注入概念名，对应 concept_root/<block>/<concept>/；仅 injection/replace 使用。",
    )
    group.add_argument(
        "--blocks",
        nargs="+",
        type=str,
        default=list(DEFAULT_INTERVENTION_BLOCKS if default_blocks is None else default_blocks),
        help="默认使用当前更优的四层配置；如需消融可显式传入。",
    )
    group.add_argument(
        "--int_mode",
        type=str,
        default="ablation",
        choices=["ablation", "injection", "replace", "projected_ablation"],
        help="概念操作模式。",
    )
    group.add_argument("--int_scale", type=float, default=5000.0, help="全局干预强度。")
    group.add_argument(
        "--int_inject_scale",
        type=float,
        default=-1.0,
        help="注入分支强度；<0 时 injection 继承 int_scale，replace 默认取 0.1 * int_scale。",
    )
    group.add_argument("--int_feature_top_k", type=int, default=10, help="从 top_positive_features.csv 读取前 K 个特征。")
    group.add_argument(
        "--int_projection_ridge",
        type=float,
        default=1e-4,
        help="projected_ablation 的岭正则强度，用于稳定子空间投影。",
    )
    group.add_argument(
        "--int_use_time_weight",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="兼容旧参数：是否使用 feature_time_scores.csv 的统计时间权重。",
    )
    group.add_argument(
        "--int_use_stat_time_weight",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否使用 feature_time_scores.csv 中的统计量时间权重。",
    )
    group.add_argument(
        "--int_stat_time_weight_scale",
        type=float,
        default=1.0,
        help="统计量时间权重的全局倍率。",
    )
    group.add_argument(
        "--int_use_learned_time_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用 SAE time_branch 产生的 learned time weight。",
    )
    group.add_argument(
        "--int_learned_time_weight_mode",
        type=str,
        default="relative_window",
        choices=["relative_window", "absolute"],
        help="learned time 的使用方式：relative_window 只学习时间形状；absolute 保留旧绝对倍率。",
    )
    group.add_argument(
        "--int_learned_time_weight_scale",
        type=float,
        default=1.0,
        help="learned time weight 的全局倍率；relative_window 下作用在平均为 1 的相对窗口上。",
    )
    group.add_argument(
        "--int_learned_time_weight_temperature",
        type=float,
        default=1.0,
        help="learned time weight 的 sigmoid 温度系数。",
    )
    group.add_argument(
        "--int_learned_time_weight_transform",
        type=str,
        default="neutral_sigmoid",
        choices=["neutral_sigmoid", "relu", "abs", "sigmoid"],
        help="absolute 模式下 learned time weight 的正权重转换方式。",
    )
    group.add_argument(
        "--int_learned_time_weight_target_mean",
        type=float,
        default=0.001,
        help="relative_window + learned_only 时使用的目标平均时间权重量级。",
    )
    group.add_argument(
        "--int_learned_time_weight_smooth_radius",
        type=int,
        default=2,
        help="relative_window 计算 raw 时间曲线时的 moving-average 半径；0 表示不平滑。",
    )
    group.add_argument(
        "--int_time_fuse_mode",
        type=str,
        default="stat_only",
        choices=["stat_only", "learned_only", "product", "sum", "max"],
        help="统计时间权重与 learned 时间权重的融合方式。",
    )
    group.add_argument(
        "--concept_dict_freq_root",
        type=str,
        default="concept_dict_freq",
        help="全局高频特征 blacklist 根目录，用于擦除前二次过滤。",
    )
    group.add_argument("--int_t_start", type=int, default=1000, help="干预时间窗上界（高噪声侧）；默认收窄到 900。")
    group.add_argument("--int_t_end", type=int, default=0, help="干预时间窗下界（低噪声侧）；默认收窄到 100。")
    group.add_argument("--int_step_start", type=int, default=-1, help=">=0 时启用 step 下界。")
    group.add_argument("--int_step_end", type=int, default=-1, help=">=0 时启用 step 上界。")
    group.add_argument(
        "--int_use_spatial_weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用空间归一化权重（按 token 范数归一化）；默认开启，关闭仅用于消融。",
    )
    group.add_argument(
        "--use_out_adapter_for_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在概念重建时把 out_adapter 也算进去；默认关闭。",
    )
    group.add_argument(
        "--int_max_delta_over_x",
        type=float,
        default=0.0,
        help=">0 时限制每个 hook step 的平均 |delta| / |x|，用于防止大批量消融直接黑图。",
    )


def build_intervention_cfg_from_args(args: argparse.Namespace) -> SharedInterventionConfig:
    """从 CLI 参数构造结构化干预配置。"""
    if float(args.int_inject_scale) < 0.0:
        mode = str(args.int_mode).lower()
        if mode == "replace":
            inject_scale = 0.1 * float(args.int_scale)
        else:
            inject_scale = float(args.int_scale)
    else:
        inject_scale = float(args.int_inject_scale)
    stat_arg = getattr(args, "int_use_stat_time_weight", None)
    compat_arg = getattr(args, "int_use_time_weight", None)
    use_stat_time_weight = bool(stat_arg if stat_arg is not None else (compat_arg if compat_arg is not None else True))
    return SharedInterventionConfig(
        mode=str(args.int_mode),
        scale=float(args.int_scale),
        inject_scale=float(inject_scale),
        feature_top_k=int(args.int_feature_top_k),
        projection_ridge=float(args.int_projection_ridge),
        use_out_adapter_for_decode=bool(args.use_out_adapter_for_decode),
        apply_only_conditional=True,
        max_delta_over_x=float(getattr(args, "int_max_delta_over_x", 0.0)),
        time=TimeInterventionConfig(
            use_stat_weight=bool(use_stat_time_weight),
            stat_weight_scale=float(getattr(args, "int_stat_time_weight_scale", 1.0)),
            use_learned_weight=bool(getattr(args, "int_use_learned_time_weight", False)),
            learned_weight_mode=str(getattr(args, "int_learned_time_weight_mode", "relative_window")),
            learned_weight_scale=float(getattr(args, "int_learned_time_weight_scale", 1.0)),
            learned_weight_temperature=float(getattr(args, "int_learned_time_weight_temperature", 1.0)),
            learned_weight_transform=str(getattr(args, "int_learned_time_weight_transform", "neutral_sigmoid")),
            learned_weight_target_mean=float(getattr(args, "int_learned_time_weight_target_mean", 0.001)),
            learned_weight_smooth_radius=int(getattr(args, "int_learned_time_weight_smooth_radius", 2)),
            fuse_mode=str(getattr(args, "int_time_fuse_mode", "stat_only")),
            t_start=int(args.int_t_start),
            t_end=int(args.int_t_end),
            step_start=_step_or_none(int(args.int_step_start)),
            step_end=_step_or_none(int(args.int_step_end)),
        ),
        spatial=SpatialInterventionConfig(
            use_norm_weight=bool(args.int_use_spatial_weight),
        ),
    )


def _build_intervention_cfg(args: argparse.Namespace) -> SharedInterventionConfig:
    """兼容旧调用路径。"""
    return build_intervention_cfg_from_args(args)


def _resolve_injectconcept(args: argparse.Namespace, *, mode: str) -> str:
    """统一解析替换/注入概念名。"""
    injectconcept = str(getattr(args, "injectconcept", "") or "").strip()
    if mode == "replace" and not injectconcept:
        raise ValueError("int_mode=replace 时必须提供 --injectconcept。")
    return injectconcept


def resolve_intervention_roots(args: argparse.Namespace) -> tuple[str, str]:
    """解析概念定位结果根目录与 blacklist 根目录。"""
    concept_root = maybe_use_sae_layout(
        path_value=str(args.concept_root),
        sae_root=str(getattr(args, "sae_root", "")),
        legacy_default="concept_dict",
        kind="concept_dig",
    )
    concept_dict_freq_root = maybe_use_sae_layout(
        path_value=str(args.concept_dict_freq_root),
        sae_root=str(getattr(args, "sae_root", "")),
        legacy_default="concept_dict_freq",
        kind="blacklist",
    )
    return concept_root, concept_dict_freq_root


def _resolve_concept_dir(*, block: str, targetconcept: str, concept_root: str) -> Path:
    """解析 SharedSAE 概念目录，并兼容旧目录结构。"""
    block_tag = block_short_name(block)
    concept_name = str(targetconcept)
    root_name = str(concept_root or "concept_dict").strip()
    candidates: List[Path] = []
    if root_name:
        root_path = Path(root_name).expanduser()
        candidates.extend(
            [
                (Path.cwd() / root_path / block_tag / concept_name).resolve(),
                (REPO_ROOT / root_path / block_tag / concept_name).resolve(),
            ]
        )
    legacy_root = Path(f"out_concept_dict_{block_tag}") / concept_name
    candidates.extend(
        [
            (Path.cwd() / legacy_root).resolve(),
            (REPO_ROOT / legacy_root).resolve(),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    tried = " | ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"概念目录不存在: tried={tried}")


def _load_topk_feature_ids(csv_path: Path, top_k: int, blacklist_ids: set[int] | None = None) -> List[int]:
    """从 top_positive_features.csv 读取前 K 个 feature_id。"""
    if int(top_k) <= 0:
        raise ValueError("feature_top_k 必须 > 0。")
    if not csv_path.exists():
        raise FileNotFoundError(f"feature rank csv 不存在: {csv_path}")

    import csv

    rows: List[Tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"feature_id", "score"}.issubset(fields):
            raise ValueError(f"rank csv 缺少列: feature_id/score, got={reader.fieldnames}")
        for row in reader:
            try:
                rows.append((int(row["feature_id"]), float(row["score"])))
            except Exception:
                continue
    rows.sort(key=lambda item: item[1], reverse=True)
    if blacklist_ids:
        blacklist = {int(fid) for fid in blacklist_ids}
        rows = [(fid, score) for fid, score in rows if int(fid) not in blacklist]
    return [feature_id for feature_id, _score in rows[: int(top_k)]]


def _resolve_feature_bundle(
    *,
    block: str,
    targetconcept: str,
    concept_root: str,
    top_k: int,
    total_steps: int,
    use_time_weight: bool,
    blacklist_root: str = "concept_dict_freq",
) -> FeatureBundle:
    """解析单个 block 的特征列表与可选时间权重。"""
    concept_dir = _resolve_concept_dir(block=block, targetconcept=targetconcept, concept_root=concept_root)
    rank_csv = concept_dir / "top_positive_features.csv"
    blacklist_ids = set(_load_blacklist_ids(str(concept_dir / "feature_blacklist.txt")))
    global_root = Path(str(blacklist_root)).expanduser()
    if not global_root.is_absolute():
        global_root = REPO_ROOT / global_root
    global_blacklist = global_root / block_short_name(block) / "feature_blacklist.txt"
    blacklist_ids.update(_load_blacklist_ids(str(global_blacklist)))
    raw_feature_ids = _load_topk_feature_ids(rank_csv, top_k)
    feature_ids = _load_topk_feature_ids(rank_csv, top_k, blacklist_ids=blacklist_ids)
    if blacklist_ids:
        print(f"[{LOG_PREFIX}] block={block} 黑名单过滤后取 topK: {len(raw_feature_ids)} -> {len(feature_ids)} (blacklist={len(blacklist_ids)})")
    if not feature_ids:
        raise ValueError(f"block={block} 过滤后 feature_ids 为空。")

    coeff_by_step: Dict[int, torch.Tensor] = {}
    if bool(use_time_weight):
        coeff_csv = concept_dir / "feature_time_scores.csv"
        coeff_raw = _load_coeff_by_step_from_exp53_csv(csv_path=str(coeff_csv), feature_ids=feature_ids)
        coeff_by_step = _interpolate_coeff_by_step(coeff_by_step=coeff_raw, total_steps=int(total_steps))

    return FeatureBundle(
        concept_name=str(targetconcept),
        feature_ids=[int(x) for x in feature_ids],
        feature_scales=[1.0 for _ in feature_ids],
        coeff_by_step=coeff_by_step,
    )


def _scale_coeff_by_step(
    coeff_by_step: Dict[int, torch.Tensor],
    *,
    scale: float,
) -> Dict[int, torch.Tensor]:
    """按需放大/缩小时间权重表。"""
    s = float(scale)
    if not coeff_by_step or abs(s - 1.0) <= 1e-12:
        return coeff_by_step
    return {int(step): vec * s for step, vec in coeff_by_step.items()}


def _stat_time_weight_for_step(
    *,
    coeff_by_step: Dict[int, torch.Tensor],
    step_idx: int,
    all_feature_ids: List[int],
    active_feature_ids: List[int],
    device,
    dtype,
) -> torch.Tensor:
    """读取当前 step 的统计时间权重，缺失位置严格置 0。"""
    out = torch.zeros(len(active_feature_ids), device=device, dtype=dtype)
    coeff_vec_full = coeff_by_step.get(int(step_idx))
    if coeff_vec_full is None or int(coeff_vec_full.numel()) != len(all_feature_ids):
        return out
    fid_to_pos = {int(fid): idx for idx, fid in enumerate(all_feature_ids)}
    coeff_on_device = coeff_vec_full.to(device=device, dtype=dtype)
    for out_pos, fid in enumerate(active_feature_ids):
        if int(fid) in fid_to_pos:
            out[int(out_pos)] = coeff_on_device[int(fid_to_pos[int(fid)])]
    return out


def _fuse_time_weights(
    *,
    stat_weight: torch.Tensor | None,
    learned_weight: torch.Tensor | None,
    mode: str,
    learned_mode: str = "absolute",
    learned_target_mean: float = 0.001,
    device,
    dtype,
    n_features: int,
) -> torch.Tensor | None:
    """融合统计时间权重与 learned time weight。"""
    if stat_weight is None and learned_weight is None:
        return None
    if stat_weight is None:
        stat_weight = torch.ones(int(n_features), device=device, dtype=dtype)
    if learned_weight is None:
        learned_weight = torch.ones(int(n_features), device=device, dtype=dtype)
    mode_norm = str(mode).strip().lower()
    learned_mode_norm = str(learned_mode).strip().lower()
    if mode_norm == "stat_only":
        return stat_weight
    if mode_norm == "learned_only":
        if learned_mode_norm == "relative_window":
            return learned_weight * float(learned_target_mean)
        return learned_weight
    if mode_norm == "product":
        return stat_weight * learned_weight
    learned_for_additive = (
        learned_weight * float(learned_target_mean) if learned_mode_norm == "relative_window" else learned_weight
    )
    if mode_norm == "sum":
        return 0.5 * (stat_weight + learned_for_additive)
    if mode_norm == "max":
        return torch.maximum(stat_weight, learned_for_additive)
    raise ValueError(f"未知 time fuse mode: {mode}")


def _smooth_time_series(values: torch.Tensor, *, radius: int) -> torch.Tensor:
    """沿 step 维做轻量 moving average。"""
    r = int(radius)
    if r <= 0 or int(values.shape[0]) <= 1:
        return values
    rows = []
    n = int(values.shape[0])
    for idx in range(n):
        lo = max(0, idx - r)
        hi = min(n, idx + r + 1)
        rows.append(values[lo:hi].mean(dim=0))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def _learned_time_weight_for_step(
    *,
    sae: SharedSAE,
    timesteps,
    step_idx: int,
    timestep_t: torch.Tensor,
    feature_ids: List[int],
    mode: str,
    transform: str,
    temperature: float,
    scale: float,
    smooth_radius: int,
    cache: Dict[Tuple[object, ...], Dict[str, torch.Tensor | float]],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """返回当前 step 的 learned raw/weight，并在 relative 模式下按全时间窗归一。"""
    device = sae.latent_bias.device
    dtype = sae.latent_bias.dtype
    ids_key = tuple(int(fid) for fid in feature_ids)
    mode_norm = str(mode).strip().lower()
    temp = max(float(temperature), 1e-6)
    scale_f = float(scale)
    radius = max(0, int(smooth_radius))

    if mode_norm == "absolute":
        raw, weight = sae.get_learned_time_weight(
            timestep=timestep_t,
            feature_ids=list(ids_key),
            transform=str(transform),
            temperature=float(temperature),
            scale=scale_f,
        )
        return raw.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype), {
            "learned_temporal_cv": 0.0,
            "learned_temporal_range": 0.0,
        }

    if timesteps is None:
        timestep_values = [float(timestep_t.reshape(-1)[0].item())]
    else:
        timestep_values = [float(t.item() if hasattr(t, "item") else t) for t in timesteps]
    if not timestep_values:
        timestep_values = [float(timestep_t.reshape(-1)[0].item())]
    safe_step = min(max(0, int(step_idx)), len(timestep_values) - 1)
    cache_key = ("relative_window", ids_key, tuple(round(x, 6) for x in timestep_values), temp, scale_f, radius)
    cached = cache.get(cache_key)
    if cached is None:
        raw_rows = []
        for t_val in timestep_values:
            raw, _weight = sae.get_learned_time_weight(
                timestep=torch.tensor([float(t_val)], device=device, dtype=dtype),
                feature_ids=list(ids_key),
                transform="neutral_sigmoid",
                temperature=1.0,
                scale=1.0,
            )
            raw_rows.append(raw.to(device=device, dtype=dtype))
        raw_tf = torch.stack(raw_rows, dim=0) if raw_rows else torch.zeros(1, len(ids_key), device=device, dtype=dtype)
        smooth = _smooth_time_series(raw_tf, radius=radius)
        centered = smooth - smooth.mean(dim=0, keepdim=True)
        std = smooth.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        raw_z = centered / std
        learned_rel = torch.sigmoid(raw_z / temp)
        learned_rel = learned_rel / learned_rel.mean(dim=0, keepdim=True).clamp_min(1e-6)
        learned_rel = learned_rel * scale_f
        mean = learned_rel.detach().float().mean().clamp_min(1e-12)
        std_all = learned_rel.detach().float().std(unbiased=False)
        cached = {
            "raw_tf": raw_tf,
            "weight_tf": learned_rel,
            "learned_temporal_cv": float((std_all / mean).item()),
            "learned_temporal_range": float((learned_rel.detach().float().max() - learned_rel.detach().float().min()).item()),
        }
        cache[cache_key] = cached

    raw_tf = cached["raw_tf"]
    weight_tf = cached["weight_tf"]
    assert isinstance(raw_tf, torch.Tensor)
    assert isinstance(weight_tf, torch.Tensor)
    stats = {
        "learned_temporal_cv": float(cached.get("learned_temporal_cv", 0.0)),
        "learned_temporal_range": float(cached.get("learned_temporal_range", 0.0)),
    }
    return raw_tf[safe_step].to(device=device, dtype=dtype), weight_tf[safe_step].to(device=device, dtype=dtype), stats


def _tensor_or_blank(values: torch.Tensor | None, idx: int) -> str | float:
    """CSV 诊断里把可选 tensor 值转成数字或空字符串。"""
    if values is None or int(idx) >= int(values.numel()):
        return ""
    return float(values.detach().float().cpu()[int(idx)].item())


def _record_time_weight_rows(
    *,
    hook,
    concept: str,
    role: str,
    block: str,
    step_idx: int,
    timestep: int,
    feature_ids: List[int],
    stat_weight: torch.Tensor | None,
    learned_raw: torch.Tensor | None,
    learned_weight: torch.Tensor | None,
    final_time_weight: torch.Tensor | None,
    coeff_base: torch.Tensor,
    coeff_final: torch.Tensor,
    debug_row: Dict[str, object],
    int_scale: float,
    learned_temporal_cv: float | str = "",
    learned_temporal_range: float | str = "",
) -> None:
    """记录 long/summary 两种时间权重诊断行。"""
    long_rows = getattr(hook, "time_weight_long_rows", None)
    summary_rows = getattr(hook, "time_weight_summary_rows", None)
    if long_rows is None or summary_rows is None:
        return

    base_abs = coeff_base.detach().abs().mean(dim=0)
    final_abs = coeff_final.detach().abs().mean(dim=0)
    for pos, fid in enumerate(feature_ids):
        long_rows.append(
            {
                "concept": str(concept),
                "role": str(role),
                "block": str(block),
                "step_idx": int(step_idx),
                "timestep": int(timestep),
                "feature_id": int(fid),
                "stat_weight": _tensor_or_blank(stat_weight, pos),
                "learned_time_raw": _tensor_or_blank(learned_raw, pos),
                "learned_weight": _tensor_or_blank(learned_weight, pos),
                "final_time_weight": _tensor_or_blank(final_time_weight, pos),
                "base_coeff_mean_abs": float(base_abs[pos].item()),
                "final_coeff_mean_abs": float(final_abs[pos].item()),
            }
        )

    def _mean_or_blank(x: torch.Tensor | None) -> str | float:
        if x is None or int(x.numel()) == 0:
            return ""
        return float(x.detach().abs().float().mean().item())

    def _max_or_blank(x: torch.Tensor | None) -> str | float:
        if x is None or int(x.numel()) == 0:
            return ""
        return float(x.detach().abs().float().max().item())

    if int(final_abs.numel()) > 0:
        top_k = min(5, int(final_abs.numel()))
        top_vals, top_pos = torch.topk(final_abs, k=top_k)
        top_ids = " ".join(str(int(feature_ids[int(pos.item())])) for pos in top_pos)
        top_weights = " ".join(f"{float(val.item()):.6g}" for val in top_vals)
    else:
        top_ids = ""
        top_weights = ""

    summary_rows.append(
        {
            "concept": str(concept),
            "role": str(role),
            "block": str(block),
            "step_idx": int(step_idx),
            "timestep": int(timestep),
            "num_features": int(len(feature_ids)),
            "stat_weight_mean": _mean_or_blank(stat_weight),
            "stat_weight_max": _max_or_blank(stat_weight),
            "learned_weight_mean": _mean_or_blank(learned_weight),
            "learned_weight_max": _max_or_blank(learned_weight),
            "final_weight_mean": _mean_or_blank(final_time_weight),
            "final_weight_max": _max_or_blank(final_time_weight),
            "int_scale": float(int_scale),
            "effective_gain_mean": (
                "" if final_time_weight is None else float(int_scale) * float(final_time_weight.detach().abs().float().mean().item())
            ),
            "effective_gain_max": (
                "" if final_time_weight is None else float(int_scale) * float(final_time_weight.detach().abs().float().max().item())
            ),
            "learned_temporal_cv": learned_temporal_cv,
            "learned_temporal_range": learned_temporal_range,
            "mean_abs_c_base": float(coeff_base.detach().abs().mean().item()),
            "mean_abs_c_final": float(coeff_final.detach().abs().mean().item()),
            "mean_abs_delta_x": float(debug_row.get("mean_abs_delta_x", 0.0) or 0.0),
            "delta_over_x": float(debug_row.get("delta_over_x", 0.0) or 0.0),
            "delta_safety_scale": float(debug_row.get("delta_safety_scale", 1.0) or 1.0),
            "top_feature_ids_final": top_ids,
            "top_feature_weights_final": top_weights,
        }
    )


@torch.no_grad()
def _decode_selected_features_norm(
    *,
    model: SharedSAE,
    x_norm: torch.Tensor,
    block_name: str,
    timestep_t: torch.Tensor,
    coords_norm: torch.Tensor,
    feature_ids: List[int],
    feature_scales: torch.Tensor,
    coeff_t: torch.Tensor | None,
    use_out_adapter_for_decode: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 SharedSAE 特征空间里解出所选特征的重建贡献。"""
    x_adapted = model._apply_input_adapter(x_norm, block_name)
    base = model.encoder(x_adapted - model.pre_bias) + model.latent_bias
    pre_act = model._compose_pre_activation(base=base, timestep=timestep_t, coords_norm=coords_norm)
    z = _topk_keep(torch.relu(pre_act), int(model.top_k))
    coeff = z[:, feature_ids]
    coeff_final = coeff if coeff_t is None else coeff * coeff_t.unsqueeze(0)
    dirs = model.decoder.weight[:, feature_ids]
    recon_norm = (coeff_final * feature_scales.unsqueeze(0)) @ dirs.t()
    if bool(use_out_adapter_for_decode):
        recon_norm = model._apply_output_adapter(recon_norm, block_name, use_out_adapter=True)
    return recon_norm, coeff, coeff_final


def _project_selected_subspace_norm(
    *,
    model: SharedSAE,
    x_norm: torch.Tensor,
    block_name: str,
    feature_ids: List[int],
    feature_scales: torch.Tensor,
    coeff_t: torch.Tensor | None,
    use_out_adapter_for_decode: bool,
    ridge: float,
) -> torch.Tensor:
    """把当前 `x_norm` 投影到所选特征张成的子空间。

    `projected_ablation` 与普通 `ablation` 的区别是：
    - 普通 ablation：减去 SAE 根据当前编码得到的概念重建
    - projected_ablation：减去 `x_norm` 在目标特征子空间上的最小二乘投影

    当选中特征方向彼此不够正交时，后者通常更接近“只删子空间，不删额外分量”。
    """
    if not feature_ids:
        return torch.zeros_like(x_norm)

    basis = model.decoder.weight[:, feature_ids].to(device=x_norm.device, dtype=x_norm.dtype)  # [d_model, k]
    if int(feature_scales.numel()) == int(basis.shape[1]):
        basis = basis * feature_scales.abs().unsqueeze(0)

    if coeff_t is not None:
        gate = coeff_t.detach().abs().to(device=x_norm.device, dtype=x_norm.dtype)
        if int(gate.numel()) == int(basis.shape[1]):
            active = gate > 1e-12
            if not bool(active.any()):
                return torch.zeros_like(x_norm)
            basis = basis[:, active]
            gate = gate[active]
            basis = basis * gate.unsqueeze(0)

    if int(basis.shape[1]) == 0:
        return torch.zeros_like(x_norm)

    # cuSOLVER 在 half/bfloat16 上的 LU 分解支持不稳定；这里固定升到 fp32
    # 做岭正则最小二乘，再把结果投回原 dtype，避免 projected_ablation
    # 在 11GB 卡上因为半精度线性求解直接报错。
    basis_solve = basis.to(dtype=torch.float32)
    x_norm_solve = x_norm.to(dtype=torch.float32)
    gram = basis_solve.t() @ basis_solve
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    gram_reg = gram + float(max(ridge, 0.0)) * eye
    rhs = x_norm_solve @ basis_solve  # [tokens, k]
    coeff_ls = torch.linalg.solve(gram_reg, rhs.t()).t()  # [tokens, k]
    proj_norm = coeff_ls @ basis_solve.t()
    proj_norm = proj_norm.to(dtype=x_norm.dtype)
    if bool(use_out_adapter_for_decode):
        proj_norm = model._apply_output_adapter(proj_norm, block_name, use_out_adapter=True)
    return proj_norm


def build_shared_feature_intervention_hook(
    *,
    pipe,
    sae: SharedSAE,
    spec: InterventionSpec,
    block_name: str,
    block_norm_scale: float,
    use_out_adapter_for_decode: bool,
):
    """构建 SharedSAE 版概念干预 hook。"""
    mode = str(spec.mode).lower()
    if mode not in {"ablation", "injection", "replace", "projected_ablation"}:
        raise ValueError(f"不支持的干预模式: {spec.mode}")

    state = {"step": 0, "debug_rows": []}
    learned_time_cache: Dict[Tuple[object, ...], Dict[str, torch.Tensor | float]] = {}
    feature_ids_raw, feature_scales_raw = _resolve_feature_list(spec)
    inject_feature_ids_raw = [int(fid) for fid in getattr(spec, "inject_feature_ids", ())]
    inject_feature_scales_raw = [float(scale) for scale in getattr(spec, "inject_feature_scales", ())]

    def hook(module, input, output):
        tensor_out, is_tuple = _extract_tensor(output)
        step_idx = int(state["step"])
        timesteps = getattr(getattr(pipe, "scheduler", None), "timesteps", None)
        t_now = -1
        if timesteps is not None and step_idx < len(timesteps):
            t_now = int(timesteps[step_idx])
        state["step"] += 1

        dbg = {
            "step_idx": int(step_idx),
            "timestep": int(t_now),
            "active": 0,
            "mode": str(mode),
            "scale": float(spec.scale),
            "mean_abs_c_base": 0.0,
            "mean_abs_w_time": 1.0,
            "mean_abs_c_final": 0.0,
            "mean_abs_recon_pre_spatial": 0.0,
            "mean_abs_recon_final": 0.0,
            "mean_abs_delta_x": 0.0,
            "delta_over_x": 0.0,
            "delta_safety_scale": 1.0,
            "active_feature_ids_time": "",
            "active_feature_ids_final": "",
            "top_feature_ids_final": "",
            "top_feature_scores_final": "",
            "inject_active_feature_ids_final": "",
            "inject_top_feature_ids_final": "",
            "inject_top_feature_scores_final": "",
            "mean_abs_recon_inject": 0.0,
            "mean_abs_delta_x_inject": 0.0,
        }

        if not _in_time_window(step_idx=step_idx, t_now=t_now, spec=spec):
            state["debug_rows"].append(dbg)
            return output
        if tensor_out.dim() not in (3, 4):
            state["debug_rows"].append(dbg)
            return output

        out = tensor_out.clone()
        sl = _conditional_slice(out, spec.apply_only_conditional)
        selected = out[sl]
        flat_out, meta = _flatten_spatial(selected)

        in_tensor = _extract_input_tensor(input)
        if in_tensor is None or in_tensor.dim() not in (3, 4):
            raise RuntimeError(
                f"[{LOG_PREFIX}] 无法提取 input tensor，不能构造 delta(out-in)。"
                f" block={block_name}, step_idx={step_idx}"
            )
        in_sel = in_tensor[sl].to(device=selected.device, dtype=selected.dtype)
        flat_in, meta_in = _flatten_spatial(in_sel)
        if meta_in != meta or flat_in.shape != flat_out.shape:
            raise RuntimeError(
                f"[{LOG_PREFIX}] input/output 形状不匹配。"
                f" block={block_name}, step_idx={step_idx}, out_meta={meta}, in_meta={meta_in},"
                f" out_shape={tuple(flat_out.shape)}, in_shape={tuple(flat_in.shape)}"
            )
        if int(flat_out.shape[-1]) != int(sae.d_model):
            state["debug_rows"].append(dbg)
            return output

        params = next(sae.parameters())
        flat_delta = (flat_out - flat_in).to(device=params.device, dtype=params.dtype)
        coords_norm = coords_from_meta(meta, device=params.device, dtype=params.dtype)
        timestep_t = torch.tensor([float(max(t_now, 0))], device=params.device, dtype=params.dtype)
        x_norm = flat_delta * float(block_norm_scale)

        n_feat = int(sae.decoder.weight.shape[1])
        ids = [int(fid) for fid in feature_ids_raw if 0 <= int(fid) < n_feat]
        if not ids:
            state["debug_rows"].append(dbg)
            return output

        id_to_scale = {int(fid): float(scale) for fid, scale in zip(feature_ids_raw, feature_scales_raw)}
        scales_t = torch.tensor([id_to_scale[int(fid)] for fid in ids], device=params.device, dtype=params.dtype)

        stat_weight = None
        use_stat_weight = bool(getattr(spec, "use_stat_time_weight", False)) or str(spec.coeff_source).lower() == "from_csv"
        if use_stat_weight:
            if not spec.coeff_by_step:
                raise RuntimeError(f"[{LOG_PREFIX}] use_stat_time_weight=True 但 coeff_by_step 为空。block={block_name}")
            stat_weight = _stat_time_weight_for_step(
                coeff_by_step=spec.coeff_by_step,
                step_idx=step_idx,
                all_feature_ids=[int(fid) for fid in feature_ids_raw],
                active_feature_ids=ids,
                device=params.device,
                dtype=params.dtype,
            )
            stat_weight = stat_weight * float(getattr(spec, "stat_time_weight_scale", spec.time_weight_scale))

        learned_raw = None
        learned_weight = None
        learned_stats: Dict[str, float] = {}
        if bool(getattr(spec, "use_learned_time_weight", False)):
            learned_raw, learned_weight, learned_stats = _learned_time_weight_for_step(
                sae=sae,
                timesteps=timesteps,
                step_idx=step_idx,
                timestep_t=timestep_t,
                feature_ids=ids,
                mode=str(getattr(spec, "learned_time_weight_mode", "absolute")),
                transform=str(getattr(spec, "learned_time_weight_transform", "neutral_sigmoid")),
                temperature=float(getattr(spec, "learned_time_weight_temperature", 1.0)),
                scale=float(getattr(spec, "learned_time_weight_scale", 1.0)),
                smooth_radius=int(getattr(spec, "learned_time_weight_smooth_radius", 2)),
                cache=learned_time_cache,
            )
            learned_raw = learned_raw.to(device=params.device, dtype=params.dtype)
            learned_weight = learned_weight.to(device=params.device, dtype=params.dtype)

        coeff_t = _fuse_time_weights(
            stat_weight=stat_weight,
            learned_weight=learned_weight,
            mode=str(getattr(spec, "time_fuse_mode", "stat_only")),
            learned_mode=str(getattr(spec, "learned_time_weight_mode", "absolute")),
            learned_target_mean=float(getattr(spec, "learned_time_weight_target_mean", 0.001)),
            device=params.device,
            dtype=params.dtype,
            n_features=len(ids),
        )

        recon_norm, coeff_base, coeff_final = _decode_selected_features_norm(
            model=sae,
            x_norm=x_norm,
            block_name=block_name,
            timestep_t=timestep_t,
            coords_norm=coords_norm,
            feature_ids=ids,
            feature_scales=scales_t,
            coeff_t=coeff_t,
            use_out_adapter_for_decode=bool(use_out_adapter_for_decode),
        )
        if mode == "projected_ablation":
            recon_norm = _project_selected_subspace_norm(
                model=sae,
                x_norm=x_norm,
                block_name=block_name,
                feature_ids=ids,
                feature_scales=scales_t,
                coeff_t=coeff_t,
                use_out_adapter_for_decode=bool(use_out_adapter_for_decode),
                ridge=float(spec.projection_ridge),
            )
        dbg["mean_abs_c_base"] = float(coeff_base.detach().abs().mean().item())
        if coeff_t is not None:
            dbg["mean_abs_w_time"] = float(coeff_t.detach().abs().mean().item())
            active_time_pos = (coeff_t.detach().abs() > 1e-12).nonzero(as_tuple=False).flatten().tolist()
            dbg["active_feature_ids_time"] = " ".join(str(int(ids[idx])) for idx in active_time_pos)
        dbg["mean_abs_c_final"] = float(coeff_final.detach().abs().mean().item())

        per_feat_abs = coeff_final.detach().abs().mean(dim=0)
        active_final_pos = (per_feat_abs > 1e-12).nonzero(as_tuple=False).flatten().tolist()
        dbg["active_feature_ids_final"] = " ".join(str(int(ids[idx])) for idx in active_final_pos)
        if int(per_feat_abs.numel()) > 0:
            top_k = min(5, int(per_feat_abs.numel()))
            top_vals, top_pos = torch.topk(per_feat_abs, k=top_k)
            dbg["top_feature_ids_final"] = " ".join(str(int(ids[int(idx.item())])) for idx in top_pos)
            dbg["top_feature_scores_final"] = " ".join(f"{float(val.item()):.6g}" for val in top_vals)

        recon = recon_norm / max(abs(float(block_norm_scale)), 1e-12)
        dbg["mean_abs_recon_pre_spatial"] = float(recon.detach().abs().mean().item())
        recon = _maybe_apply_spatial_norm_weight(
            recon=recon,
            flat=flat_out.to(device=recon.device, dtype=recon.dtype),
            meta=meta,
            spec=spec,
        )
        dbg["mean_abs_recon_final"] = float(recon.detach().abs().mean().item())

        gain = float(spec.scale)
        flat_new = flat_out.to(device=recon.device, dtype=recon.dtype)
        signed_delta = gain * recon
        if mode == "injection":
            delta_sign = 1.0
        else:
            delta_sign = -1.0
        mean_abs_delta = float(signed_delta.detach().abs().mean().item())
        mean_abs_x = float(flat_out.detach().abs().mean().item())
        delta_ratio = float(mean_abs_delta / (mean_abs_x + 1e-12))
        max_delta_over_x = float(getattr(spec, "max_delta_over_x", 0.0) or 0.0)
        safety_scale = 1.0
        if max_delta_over_x > 0.0 and delta_ratio > max_delta_over_x:
            safety_scale = float(max_delta_over_x / (delta_ratio + 1e-12))
            signed_delta = signed_delta * safety_scale
            mean_abs_delta = float(signed_delta.detach().abs().mean().item())
            delta_ratio = float(mean_abs_delta / (mean_abs_x + 1e-12))
        flat_new = flat_new + float(delta_sign) * signed_delta
        dbg["mean_abs_delta_x"] = mean_abs_delta
        dbg["delta_over_x"] = delta_ratio
        dbg["delta_safety_scale"] = safety_scale
        _record_time_weight_rows(
            hook=hook,
            concept=str(getattr(spec, "targetconcept", "") or ""),
            role="target",
            block=str(block_name),
            step_idx=int(step_idx),
            timestep=int(t_now),
            feature_ids=ids,
            stat_weight=stat_weight,
            learned_raw=learned_raw,
            learned_weight=learned_weight,
            final_time_weight=coeff_t,
            coeff_base=coeff_base,
            coeff_final=coeff_final,
            debug_row=dbg,
            int_scale=gain,
            learned_temporal_cv=learned_stats.get("learned_temporal_cv", ""),
            learned_temporal_range=learned_stats.get("learned_temporal_range", ""),
        )

        if mode == "replace":
            n_feat_inject = int(sae.decoder.weight.shape[1])
            inject_ids = [int(fid) for fid in inject_feature_ids_raw if 0 <= int(fid) < n_feat_inject]
            if not inject_ids:
                raise RuntimeError(f"[{LOG_PREFIX}] int_mode=replace 但 inject_feature_ids 为空。block={block_name}")

            inject_id_to_scale = {
                int(fid): float(scale) for fid, scale in zip(inject_feature_ids_raw, inject_feature_scales_raw)
            }
            inject_scales_t = torch.tensor(
                [inject_id_to_scale[int(fid)] for fid in inject_ids],
                device=params.device,
                dtype=params.dtype,
            )

            inject_stat_weight = None
            use_inject_stat_weight = bool(getattr(spec, "inject_use_stat_time_weight", False)) or (
                str(spec.coeff_source).lower() == "from_csv"
            )
            if use_inject_stat_weight:
                if not spec.inject_coeff_by_step:
                    raise RuntimeError(
                        f"[{LOG_PREFIX}] int_mode=replace 且 use_stat_time_weight=True，但 inject_coeff_by_step 为空。"
                        f" block={block_name}"
                    )
                inject_stat_weight = _stat_time_weight_for_step(
                    coeff_by_step=spec.inject_coeff_by_step,
                    step_idx=step_idx,
                    all_feature_ids=[int(fid) for fid in inject_feature_ids_raw],
                    active_feature_ids=inject_ids,
                    device=params.device,
                    dtype=params.dtype,
                )
                inject_stat_weight = inject_stat_weight * float(
                    getattr(spec, "inject_stat_time_weight_scale", spec.inject_time_weight_scale)
                )

            inject_learned_raw = None
            inject_learned_weight = None
            inject_learned_stats: Dict[str, float] = {}
            if bool(getattr(spec, "inject_use_learned_time_weight", getattr(spec, "use_learned_time_weight", False))):
                inject_learned_raw, inject_learned_weight, inject_learned_stats = _learned_time_weight_for_step(
                    sae=sae,
                    timesteps=timesteps,
                    step_idx=step_idx,
                    timestep_t=timestep_t,
                    feature_ids=inject_ids,
                    mode=str(getattr(spec, "learned_time_weight_mode", "absolute")),
                    transform=str(getattr(spec, "learned_time_weight_transform", "neutral_sigmoid")),
                    temperature=float(getattr(spec, "learned_time_weight_temperature", 1.0)),
                    scale=float(getattr(spec, "inject_learned_time_weight_scale", spec.learned_time_weight_scale)),
                    smooth_radius=int(getattr(spec, "learned_time_weight_smooth_radius", 2)),
                    cache=learned_time_cache,
                )
                inject_learned_raw = inject_learned_raw.to(device=params.device, dtype=params.dtype)
                inject_learned_weight = inject_learned_weight.to(device=params.device, dtype=params.dtype)

            inject_coeff_t = _fuse_time_weights(
                stat_weight=inject_stat_weight,
                learned_weight=inject_learned_weight,
                mode=str(getattr(spec, "time_fuse_mode", "stat_only")),
                learned_mode=str(getattr(spec, "learned_time_weight_mode", "absolute")),
                learned_target_mean=float(getattr(spec, "learned_time_weight_target_mean", 0.001)),
                device=params.device,
                dtype=params.dtype,
                n_features=len(inject_ids),
            )

            inject_recon_norm, _inject_coeff_base, inject_coeff_final = _decode_selected_features_norm(
                model=sae,
                x_norm=x_norm,
                block_name=block_name,
                timestep_t=timestep_t,
                coords_norm=coords_norm,
                feature_ids=inject_ids,
                feature_scales=inject_scales_t,
                coeff_t=inject_coeff_t,
                use_out_adapter_for_decode=bool(use_out_adapter_for_decode),
            )
            inject_recon = inject_recon_norm / max(abs(float(block_norm_scale)), 1e-12)
            inject_recon = _maybe_apply_spatial_norm_weight(
                recon=inject_recon,
                flat=flat_out.to(device=inject_recon.device, dtype=inject_recon.dtype),
                meta=meta,
                spec=spec,
            )
            inject_gain = float(spec.inject_scale)
            inject_delta = inject_gain * inject_recon
            inject_mean_abs_delta = float(inject_delta.detach().abs().mean().item())
            inject_ratio = float(inject_mean_abs_delta / (mean_abs_x + 1e-12))
            inject_safety_scale = 1.0
            if max_delta_over_x > 0.0 and inject_ratio > max_delta_over_x:
                inject_safety_scale = float(max_delta_over_x / (inject_ratio + 1e-12))
                inject_delta = inject_delta * inject_safety_scale
                inject_mean_abs_delta = float(inject_delta.detach().abs().mean().item())
            flat_new = flat_new + inject_delta
            dbg["mean_abs_recon_inject"] = float(inject_recon.detach().abs().mean().item())
            dbg["mean_abs_delta_x_inject"] = inject_mean_abs_delta

            inject_per_feat_abs = inject_coeff_final.detach().abs().mean(dim=0)
            inject_active_pos = (inject_per_feat_abs > 1e-12).nonzero(as_tuple=False).flatten().tolist()
            dbg["inject_active_feature_ids_final"] = " ".join(str(int(inject_ids[idx])) for idx in inject_active_pos)
            if int(inject_per_feat_abs.numel()) > 0:
                inject_top_k = min(5, int(inject_per_feat_abs.numel()))
                inject_top_vals, inject_top_pos = torch.topk(inject_per_feat_abs, k=inject_top_k)
                dbg["inject_top_feature_ids_final"] = " ".join(
                    str(int(inject_ids[int(idx.item())])) for idx in inject_top_pos
                )
                dbg["inject_top_feature_scores_final"] = " ".join(
                    f"{float(val.item()):.6g}" for val in inject_top_vals
                )
            _record_time_weight_rows(
                hook=hook,
                concept=str(getattr(spec, "injectconcept", "") or ""),
                role="inject",
                block=str(block_name),
                step_idx=int(step_idx),
                timestep=int(t_now),
                feature_ids=inject_ids,
                stat_weight=inject_stat_weight,
                learned_raw=inject_learned_raw,
                learned_weight=inject_learned_weight,
                final_time_weight=inject_coeff_t,
                coeff_base=_inject_coeff_base,
                coeff_final=inject_coeff_final,
                debug_row={
                    **dbg,
                    "mean_abs_delta_x": dbg.get("mean_abs_delta_x_inject", 0.0),
                    "delta_safety_scale": inject_safety_scale,
                    "delta_over_x": 0.0,
                },
                int_scale=inject_gain,
                learned_temporal_cv=inject_learned_stats.get("learned_temporal_cv", ""),
                learned_temporal_range=inject_learned_stats.get("learned_temporal_range", ""),
            )

        dbg["active"] = 1
        state["debug_rows"].append(dbg)

        selected_new = _unflatten_spatial(flat_new.to(dtype=selected.dtype, device=selected.device), meta)
        out[sl] = selected_new
        return _pack_tensor(out, is_tuple)

    hook.debug_rows = state["debug_rows"]  # type: ignore[attr-defined]
    hook.time_weight_long_rows = []  # type: ignore[attr-defined]
    hook.time_weight_summary_rows = []  # type: ignore[attr-defined]
    return hook


def _build_intervention_spec(
    *,
    block: str,
    features: FeatureBundle,
    inject_features: FeatureBundle | None,
    cfg: SharedInterventionConfig,
    total_steps: int,
    block_scale_map: Dict[str, float],
    inject_block_scale_map: Dict[str, float] | None,
) -> InterventionSpec:
    """把结构化配置映射成 InterventionSpec。"""
    return InterventionSpec(
        block=block,
        feature_ids=tuple(features.feature_ids),
        feature_scales=tuple(features.feature_scales),
        inject_feature_ids=tuple(() if inject_features is None else inject_features.feature_ids),
        inject_feature_scales=tuple(() if inject_features is None else inject_features.feature_scales),
        targetconcept=str(features.concept_name),
        injectconcept=str("" if inject_features is None else inject_features.concept_name),
        mode=str(cfg.mode),
        scale=float(block_scale_map[block]),
        inject_scale=float(cfg.inject_scale if inject_block_scale_map is None else inject_block_scale_map[block]),
        projection_ridge=float(cfg.projection_ridge),
        max_delta_over_x=float(cfg.max_delta_over_x),
        t_start=int(cfg.time.t_start),
        t_end=int(cfg.time.t_end),
        use_spatial_norm_weight=bool(cfg.spatial.use_norm_weight),
        coeff_source="from_csv" if bool(cfg.time.use_stat_weight) else "from_x",
        coeff_by_step=features.coeff_by_step,
        inject_coeff_by_step={} if inject_features is None else inject_features.coeff_by_step,
        use_stat_time_weight=bool(cfg.time.use_stat_weight),
        stat_time_weight_scale=float(cfg.time.stat_weight_scale),
        use_learned_time_weight=bool(cfg.time.use_learned_weight),
        learned_time_weight_mode=str(cfg.time.learned_weight_mode),
        learned_time_weight_scale=float(cfg.time.learned_weight_scale),
        learned_time_weight_temperature=float(cfg.time.learned_weight_temperature),
        learned_time_weight_transform=str(cfg.time.learned_weight_transform),
        learned_time_weight_target_mean=float(cfg.time.learned_weight_target_mean),
        learned_time_weight_smooth_radius=int(cfg.time.learned_weight_smooth_radius),
        time_fuse_mode=str(cfg.time.fuse_mode),
        time_weight_scale=float(cfg.time.stat_weight_scale),
        inject_use_stat_time_weight=bool(cfg.time.use_stat_weight),
        inject_stat_time_weight_scale=float(cfg.time.stat_weight_scale),
        inject_use_learned_time_weight=bool(cfg.time.use_learned_weight),
        inject_learned_time_weight_scale=float(cfg.time.learned_weight_scale),
        inject_time_weight_scale=float(cfg.time.stat_weight_scale),
        step_start=cfg.time.step_start,
        step_end=cfg.time.step_end if cfg.time.step_end is not None else max(0, int(total_steps) - 1),
        apply_only_conditional=bool(cfg.apply_only_conditional),
    )


def _save_eval_pair(
    *,
    output_dir: str,
    case_number: int,
    sample_name: str,
    baseline_img,
    steered_img,
) -> None:
    """导出兼容 LPIPS/FID/CLIP 的成对图片目录。"""
    stem = f"{int(case_number)}_{safe_name(sample_name)}.png"
    original_dir = Path(output_dir).expanduser().resolve() / "eval_original"
    erased_dir = Path(output_dir).expanduser().resolve() / "eval_erased"
    original_dir.mkdir(parents=True, exist_ok=True)
    erased_dir.mkdir(parents=True, exist_ok=True)
    if baseline_img is not None:
        baseline_img.save(original_dir / stem)
    if steered_img is not None:
        steered_img.save(erased_dir / stem)


def _save_time_weight_debug_csv(*, hooks: Dict[str, object], out_dir: str) -> None:
    """导出所有 block 汇总后的时间权重诊断 CSV。"""
    ensure_dir(out_dir)
    long_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for hk in hooks.values():
        long_rows.extend(getattr(hk, "time_weight_long_rows", []) or [])
        summary_rows.extend(getattr(hk, "time_weight_summary_rows", []) or [])

    if long_rows:
        long_path = Path(out_dir).expanduser().resolve() / "diag_time_weights_long.csv"
        fields = [
            "concept",
            "role",
            "block",
            "step_idx",
            "timestep",
            "feature_id",
            "stat_weight",
            "learned_time_raw",
            "learned_weight",
            "final_time_weight",
            "base_coeff_mean_abs",
            "final_coeff_mean_abs",
        ]
        with long_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in long_rows:
                writer.writerow({key: row.get(key, "") for key in fields})

    if summary_rows:
        summary_path = Path(out_dir).expanduser().resolve() / "diag_time_weights_summary.csv"
        fields = [
            "concept",
            "role",
            "block",
            "step_idx",
            "timestep",
            "num_features",
            "stat_weight_mean",
            "stat_weight_max",
            "learned_weight_mean",
            "learned_weight_max",
            "final_weight_mean",
            "final_weight_max",
            "int_scale",
            "effective_gain_mean",
            "effective_gain_max",
            "learned_temporal_cv",
            "learned_temporal_range",
            "mean_abs_c_base",
            "mean_abs_c_final",
            "mean_abs_delta_x",
            "delta_over_x",
            "delta_safety_scale",
            "top_feature_ids_final",
            "top_feature_weights_final",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({key: row.get(key, "") for key in fields})

    _save_time_weight_plots(long_rows=long_rows, summary_rows=summary_rows, out_dir=out_dir)


def _float_or_none(value: object) -> float | None:
    """把 CSV/诊断单元格安全转成 float。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _save_time_weight_plots(
    *,
    long_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    out_dir: str,
) -> None:
    """保存 feature x timestep 时间权重热图和 top feature 处理后激活图。"""
    if not long_rows:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[{LOG_PREFIX}] 跳过时间权重绘图: {exc}")
        return

    out_root = Path(out_dir).expanduser().resolve()
    plot_groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in long_rows:
        key = (str(row.get("concept", "")), str(row.get("role", "")), str(row.get("block", "")))
        plot_groups.setdefault(key, []).append(row)

    heatmap_panels = []
    for (concept, role, block), rows in sorted(plot_groups.items()):
        feature_ids = sorted({int(row["feature_id"]) for row in rows if str(row.get("feature_id", "")).strip()})
        step_ids = sorted({int(row["step_idx"]) for row in rows if str(row.get("step_idx", "")).strip()})
        if not feature_ids or not step_ids:
            continue
        feat_to_i = {fid: i for i, fid in enumerate(feature_ids)}
        step_to_j = {step: j for j, step in enumerate(step_ids)}
        arr = np.full((len(feature_ids), len(step_ids)), np.nan, dtype=np.float32)
        for row in rows:
            weight = _float_or_none(row.get("final_time_weight"))
            if weight is None:
                continue
            arr[feat_to_i[int(row["feature_id"])], step_to_j[int(row["step_idx"])]] = float(weight)
        if np.all(np.isnan(arr)):
            continue
        heatmap_panels.append((concept, role, block, feature_ids, step_ids, arr))

    if heatmap_panels:
        n_panels = len(heatmap_panels)
        fig_h = max(3.0, 2.8 * n_panels)
        fig, axes = plt.subplots(n_panels, 1, figsize=(11, fig_h), squeeze=False)
        for ax, (concept, role, block, feature_ids, step_ids, arr) in zip(axes[:, 0], heatmap_panels):
            im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis")
            ax.set_title(f"{concept} / {role} / {block}")
            ax.set_xlabel("step_idx")
            ax.set_ylabel("feature_id")
            x_ticks = list(range(len(step_ids)))
            if len(x_ticks) > 12:
                stride = max(1, len(x_ticks) // 12)
                x_ticks = x_ticks[::stride]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(step_ids[i]) for i in x_ticks], rotation=45, ha="right")
            y_ticks = list(range(len(feature_ids)))
            if len(y_ticks) > 16:
                stride = max(1, len(y_ticks) // 16)
                y_ticks = y_ticks[::stride]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(feature_ids[i]) for i in y_ticks])
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="final_time_weight")
        fig.tight_layout()
        path = out_root / "diag_time_weights_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)

    # 画“经过时间系数处理后的平均激活权重”：final_coeff_mean_abs。
    top_rows_by_group: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    for row in long_rows:
        key = (str(row.get("concept", "")), str(row.get("role", "")), str(row.get("block", "")))
        fid = int(row["feature_id"])
        val = _float_or_none(row.get("final_coeff_mean_abs"))
        if val is None:
            continue
        top_rows_by_group.setdefault(key, {})[fid] = max(float(val), top_rows_by_group.get(key, {}).get(fid, 0.0))

    activation_panels = []
    for key, fid_to_peak in sorted(top_rows_by_group.items()):
        top_fids = [fid for fid, _v in sorted(fid_to_peak.items(), key=lambda item: item[1], reverse=True)[:5]]
        rows = plot_groups.get(key, [])
        step_ids = sorted({int(row["step_idx"]) for row in rows if int(row["feature_id"]) in set(top_fids)})
        if not top_fids or not step_ids:
            continue
        curves: Dict[int, List[float]] = {fid: [0.0 for _ in step_ids] for fid in top_fids}
        step_to_j = {step: j for j, step in enumerate(step_ids)}
        for row in rows:
            fid = int(row["feature_id"])
            if fid not in curves:
                continue
            val = _float_or_none(row.get("final_coeff_mean_abs"))
            if val is None:
                continue
            curves[fid][step_to_j[int(row["step_idx"])]] = float(val)
        activation_panels.append((key, step_ids, curves))

    if activation_panels:
        n_panels = len(activation_panels)
        fig_h = max(3.0, 2.8 * n_panels)
        fig, axes = plt.subplots(n_panels, 1, figsize=(11, fig_h), squeeze=False)
        for ax, ((concept, role, block), step_ids, curves) in zip(axes[:, 0], activation_panels):
            for fid, vals in curves.items():
                ax.plot(step_ids, vals, marker="o", linewidth=1.4, markersize=3, label=str(fid))
            ax.set_title(f"Top processed activation: {concept} / {role} / {block}")
            ax.set_xlabel("step_idx")
            ax.set_ylabel("final_coeff_mean_abs")
            ax.grid(alpha=0.25)
            ax.legend(title="feature_id", fontsize=7, ncol=min(5, max(1, len(curves))))
        fig.tight_layout()
        path = out_root / "diag_top_feature_final_activation.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)


def main() -> None:
    """程序主入口。"""
    args = parse_args()
    print(
        f"[{LOG_PREFIX}] 提醒：本脚本兼容旧 concept_dict 的 CSV 结构，"
        "但旧 SAE 找出的 feature_id 不自动等于 SharedSAE 的 feature_id。"
    )

    ckpt_dir = resolve_checkpoint_dir(ckpt_dir=str(args.ckpt_dir), output_root=str(args.output_root))
    print(f"[{LOG_PREFIX}] 使用 checkpoint: {ckpt_dir}")

    device, dtype = resolve_device_dtype(str(args.device), resolve_dtype(str(args.dtype)), log_prefix=LOG_PREFIX)
    bundle = load_shared_checkpoint_bundle(ckpt_dir=ckpt_dir, device=device, dtype=dtype)
    blocks = resolve_blocks(requested_blocks=args.blocks, ckpt_cfg=bundle.config)
    norm_scale_by_block = resolve_norm_scale_by_block(
        bundle=bundle,
        blocks=blocks,
        log_prefix=LOG_PREFIX,
        warn_if_missing=True,
    )
    steps, guidance_scale, resolution = resolve_generation_hparams(args=args, ckpt_cfg=bundle.config)
    sample_name = str(args.sample_name or args.targetconcept or "sample").strip()
    intervention_cfg = _build_intervention_cfg(args)
    mode = str(intervention_cfg.mode).lower()
    injectconcept = _resolve_injectconcept(args, mode=mode)
    concept_root, concept_dict_freq_root = resolve_intervention_roots(args)

    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix=LOG_PREFIX,
    )

    features_by_block: Dict[str, FeatureBundle] = {}
    inject_features_by_block: Dict[str, FeatureBundle] = {}
    for block in blocks:
        features_by_block[str(block)] = _resolve_feature_bundle(
            block=str(block),
            targetconcept=str(args.targetconcept),
            concept_root=str(concept_root),
            top_k=int(intervention_cfg.feature_top_k),
            total_steps=int(steps),
            use_time_weight=bool(intervention_cfg.time.use_stat_weight),
            blacklist_root=str(concept_dict_freq_root),
        )
        if mode == "replace":
            inject_features_by_block[str(block)] = _resolve_feature_bundle(
                block=str(block),
                targetconcept=str(injectconcept),
                concept_root=str(concept_root),
                top_k=int(intervention_cfg.feature_top_k),
                total_steps=int(steps),
                use_time_weight=bool(intervention_cfg.time.use_stat_weight),
                blacklist_root=str(concept_dict_freq_root),
            )
    coeffs_by_block = {
        block: _scale_coeff_by_step(
            features.coeff_by_step,
            scale=float(intervention_cfg.time.stat_weight_scale) if bool(intervention_cfg.time.use_stat_weight) else 1.0,
        )
        for block, features in features_by_block.items()
    }
    inject_coeffs_by_block = {
        block: _scale_coeff_by_step(
            features.coeff_by_step,
            scale=float(intervention_cfg.time.stat_weight_scale) if bool(intervention_cfg.time.use_stat_weight) else 1.0,
        )
        for block, features in inject_features_by_block.items()
    }
    block_scale_map = _build_block_scale_map(
        blocks=[str(block) for block in blocks],
        base_scale=float(intervention_cfg.scale),
        coeffs_by_block=coeffs_by_block,
    )
    inject_block_scale_map = None
    if mode == "replace":
        inject_block_scale_map = _build_block_scale_map(
            blocks=[str(block) for block in blocks],
            base_scale=float(intervention_cfg.inject_scale),
            coeffs_by_block=inject_coeffs_by_block,
        )

    hooks = {}
    for block in blocks:
        spec = _build_intervention_spec(
            block=str(block),
            features=features_by_block[str(block)],
            inject_features=inject_features_by_block.get(str(block)),
            cfg=intervention_cfg,
            total_steps=int(steps),
            block_scale_map=block_scale_map,
            inject_block_scale_map=inject_block_scale_map,
        )
        hooks[str(block)] = build_shared_feature_intervention_hook(
            pipe=pipe,
            sae=bundle.model,
            spec=spec,
            block_name=str(block),
            block_norm_scale=float(norm_scale_by_block[str(block)]),
            use_out_adapter_for_decode=bool(intervention_cfg.use_out_adapter_for_decode),
        )

    ensure_dir(str(args.output_dir))
    baseline_img = None
    if not bool(args.no_baseline):
        baseline_out = pipe(
            prompt=str(args.prompt),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=make_generator(int(args.seed)),
            output_type="pil",
            height=int(resolution),
            width=int(resolution),
        )
        baseline_img = extract_first_image(baseline_out)
        if baseline_img is not None:
            baseline_path = Path(args.output_dir).expanduser().resolve() / "intervention_baseline.png"
            baseline_img.save(baseline_path)
            print(f"[{LOG_PREFIX}] 已保存 baseline: {baseline_path}")

    steered_out = pipe.run_with_hooks(
        prompt=str(args.prompt),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=make_generator(int(args.seed)),
        output_type="pil",
        height=int(resolution),
        width=int(resolution),
        position_hook_dict=hooks,
    )
    steered_img = extract_first_image(steered_out)
    if steered_img is not None:
        steered_path = Path(args.output_dir).expanduser().resolve() / "intervention_steered.png"
        steered_img.save(steered_path)
        print(f"[{LOG_PREFIX}] 已保存 steered: {steered_path}")

    if baseline_img is not None and steered_img is not None:
        from PIL import Image

        compare = Image.new("RGB", (baseline_img.width + steered_img.width, max(baseline_img.height, steered_img.height)))
        compare.paste(baseline_img, (0, 0))
        compare.paste(steered_img, (baseline_img.width, 0))
        compare_path = Path(args.output_dir).expanduser().resolve() / "intervention_compare.png"
        compare.save(compare_path)
        print(f"[{LOG_PREFIX}] 已保存 compare: {compare_path}")

    _save_hook_debug_csv(hooks=hooks, out_dir=str(args.output_dir), tag="shared_intervention")
    _save_time_weight_debug_csv(hooks=hooks, out_dir=str(args.output_dir))
    _save_eval_pair(
        output_dir=str(args.output_dir),
        case_number=int(args.case_number),
        sample_name=str(sample_name),
        baseline_img=baseline_img,
        steered_img=steered_img,
    )

    manifest = {
        "ckpt_dir": str(bundle.ckpt_dir),
        "sae_root": str(getattr(args, "sae_root", "") or ""),
        "concept_root": str(concept_root),
        "concept_dict_freq_root": str(concept_dict_freq_root),
        "prompt": str(args.prompt),
        "targetconcept": str(args.targetconcept),
        "injectconcept": str(injectconcept),
        "blocks": [str(block) for block in blocks],
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "resolution": int(resolution),
        "seed": int(args.seed),
        "no_baseline": bool(args.no_baseline),
        "intervention": asdict(intervention_cfg),
        "norm_scale_by_block": norm_scale_by_block,
        "block_scale_map": block_scale_map,
        "inject_block_scale_map": {} if inject_block_scale_map is None else inject_block_scale_map,
        "features_by_block": {
            block: features.to_manifest() for block, features in features_by_block.items()
        },
        "inject_features_by_block": {
            block: features.to_manifest() for block, features in inject_features_by_block.items()
        },
        "eval_original_dir": str(Path(args.output_dir).expanduser().resolve() / "eval_original"),
        "eval_erased_dir": str(Path(args.output_dir).expanduser().resolve() / "eval_erased"),
    }
    manifest_path = Path(args.output_dir).expanduser().resolve() / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[{LOG_PREFIX}] 已保存 manifest: {manifest_path}")


if __name__ == "__main__":
    main()
