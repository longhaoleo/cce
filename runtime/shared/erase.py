"""SharedSAE 版概念擦除正式入口。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
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

    use_weight: bool = True
    weight_scale: float = 1.0
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
    feature_top_k: int = 5
    projection_ridge: float = 1e-4
    use_out_adapter_for_decode: bool = False
    apply_only_conditional: bool = True
    time: TimeInterventionConfig = field(default_factory=TimeInterventionConfig)
    spatial: SpatialInterventionConfig = field(default_factory=SpatialInterventionConfig)


@dataclass
class FeatureBundle:
    """单个 block 上待干预的特征集合。"""

    feature_ids: List[int]
    feature_scales: List[float]
    coeff_by_step: Dict[int, torch.Tensor]

    def to_manifest(self) -> Dict[str, object]:
        """导出 manifest 所需的轻量结构。"""
        return {
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
        choices=["ablation", "projected_ablation"],
        help="概念操作模式。",
    )
    group.add_argument("--int_scale", type=float, default=3000.0, help="全局干预强度。")
    group.add_argument("--int_feature_top_k", type=int, default=5, help="从 top_positive_features.csv 读取前 K 个特征。")
    group.add_argument(
        "--int_projection_ridge",
        type=float,
        default=1e-4,
        help="projected_ablation 的岭正则强度，用于稳定子空间投影。",
    )
    group.add_argument(
        "--int_use_time_weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否乘上 feature_time_scores.csv 的 step 权重。",
    )
    group.add_argument(
        "--concept_dict_freq_root",
        type=str,
        default="concept_dict_freq",
        help="全局高频特征 blacklist 根目录，用于擦除前二次过滤。",
    )
    group.add_argument("--int_t_start", type=int, default=950, help="干预时间窗上界（高噪声侧）；默认收窄到 900。")
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


def build_intervention_cfg_from_args(args: argparse.Namespace) -> SharedInterventionConfig:
    """从 CLI 参数构造结构化干预配置。"""
    return SharedInterventionConfig(
        mode=str(args.int_mode),
        scale=float(args.int_scale),
        feature_top_k=int(args.int_feature_top_k),
        projection_ridge=float(args.int_projection_ridge),
        use_out_adapter_for_decode=bool(args.use_out_adapter_for_decode),
        apply_only_conditional=True,
        time=TimeInterventionConfig(
            use_weight=bool(args.int_use_time_weight),
            weight_scale=1.0,
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
    if mode not in {"ablation", "projected_ablation"}:
        raise ValueError(f"不支持的干预模式: {spec.mode}")

    state = {"step": 0, "debug_rows": []}
    feature_ids_raw, feature_scales_raw = _resolve_feature_list(spec)

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
            "active_feature_ids_time": "",
            "active_feature_ids_final": "",
            "top_feature_ids_final": "",
            "top_feature_scores_final": "",
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

        coeff_t = None
        if str(spec.coeff_source).lower() == "from_csv":
            if not spec.coeff_by_step:
                raise RuntimeError(f"[{LOG_PREFIX}] coeff_source=from_csv 但 coeff_by_step 为空。block={block_name}")
            coeff_t = torch.zeros(len(ids), device=params.device, dtype=params.dtype)
            coeff_vec_full = spec.coeff_by_step.get(int(step_idx))
            if coeff_vec_full is not None and int(coeff_vec_full.numel()) == len(feature_ids_raw):
                fid_to_pos = {int(fid): idx for idx, fid in enumerate(feature_ids_raw)}
                pos = [fid_to_pos[int(fid)] for fid in ids if int(fid) in fid_to_pos]
                if pos:
                    coeff_t = coeff_vec_full.to(device=params.device, dtype=params.dtype)[pos]
            coeff_t = coeff_t * float(spec.time_weight_scale)

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
        flat_new = flat_out.to(device=recon.device, dtype=recon.dtype) - gain * recon
        mean_abs_delta = float((gain * recon).detach().abs().mean().item())
        mean_abs_x = float(flat_out.detach().abs().mean().item())
        dbg["mean_abs_delta_x"] = mean_abs_delta
        dbg["delta_over_x"] = float(mean_abs_delta / (mean_abs_x + 1e-12))
        dbg["active"] = 1
        state["debug_rows"].append(dbg)

        selected_new = _unflatten_spatial(flat_new.to(dtype=selected.dtype, device=selected.device), meta)
        out[sl] = selected_new
        return _pack_tensor(out, is_tuple)

    hook.debug_rows = state["debug_rows"]  # type: ignore[attr-defined]
    return hook


def _build_intervention_spec(
    *,
    block: str,
    features: FeatureBundle,
    cfg: SharedInterventionConfig,
    total_steps: int,
    block_scale_map: Dict[str, float],
) -> InterventionSpec:
    """把结构化配置映射成 InterventionSpec。"""
    return InterventionSpec(
        block=block,
        feature_ids=tuple(features.feature_ids),
        feature_scales=tuple(features.feature_scales),
        mode=str(cfg.mode),
        scale=float(block_scale_map[block]),
        projection_ridge=float(cfg.projection_ridge),
        t_start=int(cfg.time.t_start),
        t_end=int(cfg.time.t_end),
        use_spatial_norm_weight=bool(cfg.spatial.use_norm_weight),
        coeff_source="from_csv" if bool(cfg.time.use_weight) else "from_x",
        coeff_by_step=features.coeff_by_step,
        time_weight_scale=float(cfg.time.weight_scale),
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

    pipe = load_hooked_pipeline(
        model_id=str(args.model_id),
        model_local_dir=str(args.model_local_dir),
        local_files_only=bool(args.local_files_only),
        device=device,
        dtype=dtype,
        log_prefix=LOG_PREFIX,
    )

    features_by_block: Dict[str, FeatureBundle] = {}
    for block in blocks:
        features_by_block[str(block)] = _resolve_feature_bundle(
            block=str(block),
            targetconcept=str(args.targetconcept),
            concept_root=str(args.concept_root),
            top_k=int(intervention_cfg.feature_top_k),
            total_steps=int(steps),
            use_time_weight=bool(intervention_cfg.time.use_weight),
            blacklist_root=str(args.concept_dict_freq_root),
        )
    coeffs_by_block = {
        block: _scale_coeff_by_step(
            features.coeff_by_step,
            scale=float(intervention_cfg.time.weight_scale) if bool(intervention_cfg.time.use_weight) else 1.0,
        )
        for block, features in features_by_block.items()
    }
    block_scale_map = _build_block_scale_map(
        blocks=[str(block) for block in blocks],
        base_scale=float(intervention_cfg.scale),
        coeffs_by_block=coeffs_by_block,
    )

    hooks = {}
    for block in blocks:
        spec = _build_intervention_spec(
            block=str(block),
            features=features_by_block[str(block)],
            cfg=intervention_cfg,
            total_steps=int(steps),
            block_scale_map=block_scale_map,
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
    _save_eval_pair(
        output_dir=str(args.output_dir),
        case_number=int(args.case_number),
        sample_name=str(sample_name),
        baseline_img=baseline_img,
        steered_img=steered_img,
    )

    manifest = {
        "ckpt_dir": str(bundle.ckpt_dir),
        "concept_root": str(args.concept_root),
        "prompt": str(args.prompt),
        "targetconcept": str(args.targetconcept),
        "blocks": [str(block) for block in blocks],
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "resolution": int(resolution),
        "seed": int(args.seed),
        "no_baseline": bool(args.no_baseline),
        "intervention": asdict(intervention_cfg),
        "norm_scale_by_block": norm_scale_by_block,
        "block_scale_map": block_scale_map,
        "features_by_block": {
            block: features.to_manifest() for block, features in features_by_block.items()
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
