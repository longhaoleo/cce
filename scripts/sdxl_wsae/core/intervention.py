"""特征干预工具：Injection / Ablation Hook。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class InterventionSpec:
    """单个特征干预配置。"""

    block: str
    feature_ids: Tuple[int, ...] = ()
    feature_scales: Tuple[float, ...] = ()
    mode: str = "injection"  # injection | ablation
    scale: float = 1.0
    t_start: int = 600
    t_end: int = 200
    # 空间约束：默认不启用，避免改变老实验行为
    spatial_mask: str = "none"  # none | gaussian_center
    mask_sigma: float = 0.25  # sigma 的相对尺度（sigma_px = sigma * min(H,W)）
    # 系数来源：
    # - from_x: 用 z=sae.encode(x) 得到 token 级系数 c_i(x)
    # - from_csv: 用外部统计好的“按 step 的系数表”（来自 exp53 输出）
    coeff_source: str = "from_x"  # from_x | from_csv
    coeff_by_step: Dict[int, torch.Tensor] = field(default_factory=dict)  # step_idx -> [k] 系数（按 feature_ids 对齐）
    step_start: Optional[int] = None
    step_end: Optional[int] = None
    apply_only_conditional: bool = True


def _gaussian_center_mask(
    *,
    h: int,
    w: int,
    sigma_frac: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """生成以中心为峰值的 2D 高斯 mask（最大值归一化到 1）。

    返回 shape: [h*w]（按 row-major 展平）。
    """
    hh, ww = int(h), int(w)
    if hh <= 0 or ww <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    s = float(sigma_frac)
    if s <= 0:
        return torch.ones(hh * ww, device=device, dtype=dtype)
    sigma_px = s * float(min(hh, ww))
    if sigma_px <= 1e-6:
        return torch.ones(hh * ww, device=device, dtype=dtype)

    ys = torch.arange(hh, device=device, dtype=dtype)
    xs = torch.arange(ww, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy = (float(hh) - 1.0) / 2.0
    cx = (float(ww) - 1.0) / 2.0
    dy2 = (yy - cy) ** 2
    dx2 = (xx - cx) ** 2
    m = torch.exp(-(dx2 + dy2) / (2.0 * (sigma_px**2)))
    m = m / (m.max() + 1e-12)
    return m.reshape(hh * ww)


def _maybe_apply_spatial_mask(
    *,
    recon: torch.Tensor,  # [tokens, d_model]
    meta: Tuple[str, int, int, int],
    spec: InterventionSpec,
) -> torch.Tensor:
    """对 token 级 recon 乘上空间 mask（如果启用）。"""
    if str(spec.spatial_mask).lower() in ("", "none", "off", "false", "0"):
        return recon
    if str(spec.spatial_mask).lower() not in ("gaussian_center", "gaussian"):
        return recon

    kind, a, b, c = meta
    device = recon.device
    dtype = recon.dtype
    if kind == "bchw":
        bsz, h, w = int(a), int(b), int(c)
        m_hw = _gaussian_center_mask(h=h, w=w, sigma_frac=float(spec.mask_sigma), device=device, dtype=dtype)  # [h*w]
        if int(m_hw.numel()) != int(h * w) or int(recon.shape[0]) != int(bsz * h * w):
            return recon
        m = m_hw.repeat(bsz).unsqueeze(1)  # [tokens, 1]
        return recon * m

    if kind == "bnc":
        bsz, n = int(a), int(b)
        # 尝试把 token 还原成方形网格；不满足则不加 mask（避免误用）
        side = int(round(float(n) ** 0.5))
        if side > 0 and side * side == n:
            m_hw = _gaussian_center_mask(h=side, w=side, sigma_frac=float(spec.mask_sigma), device=device, dtype=dtype)
            m = m_hw.repeat(bsz).unsqueeze(1)  # [tokens, 1]
            if int(m.shape[0]) == int(recon.shape[0]):
                return recon * m
        return recon

    return recon


def _resolve_feature_list(spec: InterventionSpec) -> Tuple[List[int], List[float]]:
    """把 spec 的单特征/多特征参数统一成列表形式，保持向后兼容。"""
    ids = [int(x) for x in spec.feature_ids]
    if not ids:
        raise ValueError("feature_ids 不能为空。")

    if spec.feature_scales:
        scales = [float(x) for x in spec.feature_scales]
        if len(scales) != len(ids):
            raise ValueError("feature_scales 长度必须与 feature_ids 相同。")
    else:
        scales = [1.0 for _ in ids]

    return ids, scales


def _extract_tensor(output: Any) -> Tuple[torch.Tensor, bool]:
    """
    从 hook output 提取张量。

    返回:
    - tensor: 实际张量
    - is_tuple: 原始 output 是否是 1 元 tuple（便于恢复输出格式）
    """
    if isinstance(output, tuple):
        if len(output) != 1 or not isinstance(output[0], torch.Tensor):
            raise ValueError("当前仅支持 output 为 tensor 或长度为 1 的 tensor tuple。")
        return output[0], True
    if isinstance(output, torch.Tensor):
        return output, False
    raise ValueError("不支持的 output 类型，无法进行干预。")


def _pack_tensor(tensor: torch.Tensor, is_tuple: bool):
    """按原始格式恢复 hook 输出。"""
    if is_tuple:
        return (tensor,)
    return tensor


def _flatten_spatial(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[str, int, int, int]]:
    """
    空间展平为 token 表示。

    返回:
    - flat: [n_tokens, d_model]
    - meta: 反变换所需信息
    """
    if x.dim() == 4:
        b, c, h, w = map(int, x.shape)
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return flat, ("bchw", b, h, w)
    if x.dim() == 3:
        b, n, c = map(int, x.shape)
        flat = x.reshape(b * n, c)
        return flat, ("bnc", b, n, c)
    raise ValueError(f"不支持的输出形状: {tuple(x.shape)}")


def _unflatten_spatial(flat: torch.Tensor, meta: Tuple[str, int, int, int]) -> torch.Tensor:
    """将 token 表示恢复为原始输出形状。"""
    kind, a, b, c = meta
    if kind == "bchw":
        bsz, h, w = a, b, c
        channels = int(flat.shape[-1])
        return flat.reshape(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
    if kind == "bnc":
        bsz, n, channels = a, b, c
        _ = channels
        return flat.reshape(bsz, n, int(flat.shape[-1])).contiguous()
    raise ValueError(f"未知 meta: {meta}")


def _conditional_slice(x: torch.Tensor, only_cond: bool) -> slice:
    """返回需要应用干预的 batch 切片。"""
    if not only_cond:
        return slice(0, int(x.shape[0]))
    b = int(x.shape[0])
    if b == 2:
        return slice(1, 2)
    if b > 2 and b % 2 == 0:
        return slice(b // 2, b)
    return slice(0, b)


def _in_time_window(
    *,
    step_idx: int,
    t_now: int,
    spec: InterventionSpec,
) -> bool:
    """判断当前 hook 时刻是否在干预窗口内。"""
    # step 窗口（如果用户提供了 step 边界，优先生效；允许只给一边）
    if spec.step_start is not None:
        if step_idx < int(spec.step_start):
            return False
    if spec.step_end is not None:
        if step_idx > int(spec.step_end):
            return False

    lo_t, hi_t = sorted([int(spec.t_start), int(spec.t_end)])
    if t_now >= 0 and not (lo_t <= t_now <= hi_t):
        return False
    return True


def build_feature_intervention_hook(
    *,
    pipe: Any,
    sae: torch.nn.Module,
    spec: InterventionSpec,
):
    """
    构建可注册到 block 的前向 hook。

    干预规则：
    - injection: x <- x + scale * (c_i * d_i)
    - ablation:  x <- x - scale * (c_i * d_i)

    其中 c_i 来自当前 step、当前 token 上对 x 的 SAE 编码 `z = sae.encode(x)`。
    这样 injection/ablation 是严格对称的（只差一个符号），更像“沿特征方向加速/减速”。
    """
    mode = spec.mode.lower()
    if mode not in {"injection", "ablation"}:
        raise ValueError(f"不支持的干预模式: {spec.mode}")

    state = {"step": 0}
    feature_ids, feature_scales = _resolve_feature_list(spec)

    def hook(module, input, output):
        tensor_out, is_tuple = _extract_tensor(output)
        step_idx = int(state["step"])
        timesteps = getattr(getattr(pipe, "scheduler", None), "timesteps", None)
        t_now = -1
        if timesteps is not None and step_idx < len(timesteps):
            t_now = int(timesteps[step_idx])
        state["step"] += 1

        if not _in_time_window(step_idx=step_idx, t_now=t_now, spec=spec):
            return output

        if tensor_out.dim() not in (3, 4):
            return output

        out = tensor_out.clone()
        sl = _conditional_slice(out, spec.apply_only_conditional)
        selected = out[sl]
        flat, meta = _flatten_spatial(selected)

        d_model = int(getattr(sae, "d_model"))
        if int(flat.shape[-1]) != d_model:
            return output

        p = next(sae.parameters())
        flat = flat.to(device=p.device, dtype=p.dtype)

        # 先用 decoder 的列数作为 n_features（避免为了拿 shape 而强制 encode）
        n_feat = int(getattr(getattr(sae, "decoder", None), "weight").shape[1])
        ids = [fid for fid in feature_ids if 0 <= int(fid) < n_feat]
        if not ids:
            return output

        # 对齐 scales（保持与 ids 一一对应）
        id_to_scale = {int(fid): float(s) for fid, s in zip(feature_ids, feature_scales)}
        scales = torch.tensor([id_to_scale[int(fid)] for fid in ids], device=flat.device, dtype=flat.dtype)  # [k]

        # 取 decoder 方向矩阵：dirs shape [d_model, k]
        dirs = sae.decoder.weight[:, ids].to(device=flat.device, dtype=flat.dtype)

        g = float(spec.scale)  # 全局强度

        # 系数来源：
        # - from_x: 用 SAE.encode(flat) 得到 token 级 c_i(x)
        # - from_csv: 用预先统计好的“按 step 的系数表”，对所有 token 使用同一组系数
        src = str(spec.coeff_source).lower()
        if src == "from_csv":
            # 约定：coeff_by_step 里的向量与 spec.feature_ids 的顺序一致（长度=len(feature_ids)）
            coeff_vec_full = spec.coeff_by_step.get(int(step_idx))
            if coeff_vec_full is None:
                return output
            if int(coeff_vec_full.numel()) != len(feature_ids):
                return output

            fid_to_pos = {int(fid): i for i, fid in enumerate(feature_ids)}
            pos = [fid_to_pos[int(fid)] for fid in ids if int(fid) in fid_to_pos]
            if not pos:
                return output

            coeff_vec = coeff_vec_full.to(device=flat.device, dtype=flat.dtype)[pos]  # [k]
            coeff = coeff_vec.unsqueeze(0).expand(int(flat.shape[0]), -1)  # [tokens, k]
        else:
            z = sae.encode(flat)  # [tokens, n_features]
            coeff = z[:, ids]  # [tokens, k]

        # recon = sum_j (c_j * scale_j * d_j)
        weighted = coeff * scales.unsqueeze(0)  # [tokens, k]
        recon = weighted @ dirs.t()  # [tokens, d_model]
        recon = _maybe_apply_spatial_mask(recon=recon, meta=meta, spec=spec)

        if mode == "injection":
            flat_new = flat + g * recon
        else:
            flat_new = flat - g * recon

        selected_new = _unflatten_spatial(flat_new.to(dtype=selected.dtype, device=selected.device), meta)
        out[sl] = selected_new
        return _pack_tensor(out, is_tuple)

    return hook
