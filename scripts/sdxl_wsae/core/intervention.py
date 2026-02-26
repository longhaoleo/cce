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
    use_spatial_norm_weight: bool = False  # 按 token 范数生成空间归一化权重
    # 系数来源：
    # - from_x: 仅使用 z=sae.encode(x) 得到 token 级系数 c_i(x)
    # - from_csv: 在 from_x 的基础上再乘一个按 step 的时间权重 w_i(step)
    coeff_source: str = "from_x"  # from_x | from_csv
    coeff_by_step: Dict[int, torch.Tensor] = field(default_factory=dict)  # step_idx -> [k] 时间权重（按 feature_ids 对齐）
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
        # 归一化到均值约为 1：只做空间重分布，不额外改变全局强度
        m_hw = m_hw / (m_hw.mean() + 1e-12)
        m = m_hw.repeat(bsz).unsqueeze(1)  # [tokens, 1]
        return recon * m

    if kind == "bnc":
        bsz, n = int(a), int(b)
        # 尝试把 token 还原成方形网格；不满足则不加 mask（避免误用）
        side = int(round(float(n) ** 0.5))
        if side > 0 and side * side == n:
            m_hw = _gaussian_center_mask(h=side, w=side, sigma_frac=float(spec.mask_sigma), device=device, dtype=dtype)
            m_hw = m_hw / (m_hw.mean() + 1e-12)
            m = m_hw.repeat(bsz).unsqueeze(1)  # [tokens, 1]
            if int(m.shape[0]) == int(recon.shape[0]):
                return recon * m
        return recon

    return recon


def _maybe_apply_spatial_norm_weight(
    *,
    recon: torch.Tensor,  # [tokens, d_model]
    flat: torch.Tensor,  # [tokens, d_model]，用于计算每个 token 的范数
    meta: Tuple[str, int, int, int],
    spec: InterventionSpec,
) -> torch.Tensor:
    """按 token 范数应用空间归一化权重（每个样本内归一化）。

    设计目标：
    - 改变“空间分布”（哪些 token 更强）
    - 避免极端峰值导致图像崩溃（如整图发黑）
    """
    if not bool(getattr(spec, "use_spatial_norm_weight", False)):
        return recon

    kind, a, b, c = meta
    norms = torch.norm(flat.detach(), dim=1) + 1e-12  # [tokens]

    def _stable_spatial_weight(x_2d: torch.Tensor) -> torch.Tensor:
        """把原始 norm 映射为稳定权重：压缩动态范围 + 限幅 + 均值归一。"""
        # x_2d: [bsz, n]
        x = x_2d / (x_2d.mean(dim=1, keepdim=True) + 1e-12)
        x = torch.sqrt(torch.clamp(x, min=1e-12))  # 压缩长尾，避免尖峰
        x = torch.clamp(x, min=0.5, max=2.0)  # 限制局部增强/抑制倍率
        x = x / (x.mean(dim=1, keepdim=True) + 1e-12)  # 回到均值约 1
        return x

    if kind == "bchw":
        bsz, h, w = int(a), int(b), int(c)
        n = int(h * w)
        if bsz <= 0 or n <= 0 or int(norms.numel()) != int(bsz * n):
            return recon
        x = _stable_spatial_weight(norms.reshape(bsz, n))
        rec = recon.reshape(bsz, n, int(recon.shape[-1]))
        rec_w = rec * x.unsqueeze(-1)
        return rec_w.reshape(bsz * n, int(recon.shape[-1]))

    if kind == "bnc":
        bsz, n = int(a), int(b)
        if bsz <= 0 or n <= 0 or int(norms.numel()) != int(bsz * n):
            return recon
        x = _stable_spatial_weight(norms.reshape(bsz, n))
        rec = recon.reshape(bsz, n, int(recon.shape[-1]))
        rec_w = rec * x.unsqueeze(-1)
        return rec_w.reshape(bsz * n, int(recon.shape[-1]))

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


def _extract_input_tensor(input_obj: Any) -> Optional[torch.Tensor]:
    """从 hook input 提取主张量（通常是 input[0]）。失败时返回 None。"""
    if isinstance(input_obj, tuple) and len(input_obj) >= 1 and isinstance(input_obj[0], torch.Tensor):
        return input_obj[0]
    if isinstance(input_obj, torch.Tensor):
        return input_obj
    return None


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

    干预规则（均为“激活依赖”而非常数减法）：
    - injection: x <- x + scale * (c_i * d_i)
    - ablation:  x <- x - scale * (c_i * d_i)

    其中 c_i 来自当前 step、当前 token 上对 x 的 SAE 编码 `z = sae.encode(x)`；
    若 coeff_source=from_csv，则再乘时间权重 w_i(step)，即 c_i <- c_i * w_i(step)。
    这样 injection/ablation 是严格对称的（只差一个符号），更像“沿特征方向加速/减速”。
    """
    mode = spec.mode.lower()
    if mode not in {"injection", "ablation"}:
        raise ValueError(f"不支持的干预模式: {spec.mode}")

    state = {"step": 0, "debug_rows": []}
    feature_ids, feature_scales = _resolve_feature_list(spec)

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
        flat, meta = _flatten_spatial(selected)
        # 强制使用 delta_flat = out - in（与 exp53 的 delta 特征空间严格对齐）
        flat_delta = None
        in_tensor = _extract_input_tensor(input)
        if in_tensor is None or in_tensor.dim() not in (3, 4):
            raise RuntimeError(
                f"[intervention] 无法提取 input tensor，不能构造 delta(out-in)。"
                f" block={spec.block}, step_idx={step_idx}"
            )
        try:
            in_sel = in_tensor[sl].to(device=selected.device, dtype=selected.dtype)
            flat_in, meta_in = _flatten_spatial(in_sel)
        except Exception as e:
            raise RuntimeError(
                f"[intervention] input 展平失败，不能构造 delta(out-in)。"
                f" block={spec.block}, step_idx={step_idx}, err={e}"
            ) from e
        if meta_in != meta or flat_in.shape != flat.shape:
            raise RuntimeError(
                f"[intervention] input/output 形状不匹配，不能构造 delta(out-in)。"
                f" block={spec.block}, step_idx={step_idx}, out_meta={meta}, in_meta={meta_in},"
                f" out_shape={tuple(flat.shape)}, in_shape={tuple(flat_in.shape)}"
            )
        flat_delta = flat - flat_in

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

        # 基础系数严格来自当前 delta（out-in），不允许回退到 out。
        coeff_src = flat_delta
        coeff_src = coeff_src.to(device=p.device, dtype=p.dtype)
        z = sae.encode(coeff_src)  # [tokens, n_features]
        coeff = z[:, ids]  # [tokens, k]
        dbg["mean_abs_c_base"] = float(coeff.detach().abs().mean().item())

        # 可选：叠加按 step 的时间权重（来自 exp53）
        src = str(spec.coeff_source).lower()
        coeff_t = None
        if src == "from_csv":
            # 严格模式：只在 CSV 里“出现”的 step/feature 生效，其他全部置 0
            # 这样可避免“CSV 未覆盖区域”仍然发生干预。
            if not spec.coeff_by_step:
                raise RuntimeError(
                    f"[intervention] coeff_source=from_csv 但 coeff_by_step 为空。block={spec.block}"
                )
            coeff_t = torch.zeros(len(ids), device=flat.device, dtype=flat.dtype)  # [k]
            coeff_vec_full = spec.coeff_by_step.get(int(step_idx))
            if coeff_vec_full is not None and int(coeff_vec_full.numel()) == len(feature_ids):
                fid_to_pos = {int(fid): i for i, fid in enumerate(feature_ids)}
                pos = [fid_to_pos[int(fid)] for fid in ids if int(fid) in fid_to_pos]
                if pos:
                    coeff_t = coeff_vec_full.to(device=flat.device, dtype=flat.dtype)[pos]  # [k]
            coeff = coeff * coeff_t.unsqueeze(0)  # [tokens, k]
        if coeff_t is not None:
            dbg["mean_abs_w_time"] = float(coeff_t.detach().abs().mean().item())
            active_time_pos = (coeff_t.detach().abs() > 1e-12).nonzero(as_tuple=False).flatten().tolist()
            dbg["active_feature_ids_time"] = " ".join(str(int(ids[p])) for p in active_time_pos)
        dbg["mean_abs_c_final"] = float(coeff.detach().abs().mean().item())
        # 记录本步真正“有激活”的特征 index（按 token 平均绝对激活）
        per_feat_abs = coeff.detach().abs().mean(dim=0)  # [k]
        active_final_pos = (per_feat_abs > 1e-12).nonzero(as_tuple=False).flatten().tolist()
        dbg["active_feature_ids_final"] = " ".join(str(int(ids[p])) for p in active_final_pos)
        if int(per_feat_abs.numel()) > 0:
            tk = min(5, int(per_feat_abs.numel()))
            top_vals, top_pos = torch.topk(per_feat_abs, k=tk)
            dbg["top_feature_ids_final"] = " ".join(str(int(ids[int(p.item())])) for p in top_pos)
            dbg["top_feature_scores_final"] = " ".join(f"{float(v.item()):.6g}" for v in top_vals)

        # recon = sum_j (c_j * scale_j * d_j)
        # 即干预始终是 activation-dependent：
        # x_new = x + beta * f_c(x) * W_dec,c
        # 这里 beta 对应 mode 与全局 scale（injection:+g, ablation:-g）。
        weighted = coeff * scales.unsqueeze(0)  # [tokens, k]
        recon = weighted @ dirs.t()  # [tokens, d_model]
        dbg["mean_abs_recon_pre_spatial"] = float(recon.detach().abs().mean().item())
        recon = _maybe_apply_spatial_norm_weight(recon=recon, flat=flat, meta=meta, spec=spec)
        recon = _maybe_apply_spatial_mask(recon=recon, meta=meta, spec=spec)
        dbg["mean_abs_recon_final"] = float(recon.detach().abs().mean().item())

        if mode == "injection":
            flat_new = flat + g * recon
        else:
            flat_new = flat - g * recon
        mean_abs_delta = float((g * recon).detach().abs().mean().item())
        mean_abs_x = float(flat.detach().abs().mean().item())
        dbg["mean_abs_delta_x"] = mean_abs_delta
        dbg["delta_over_x"] = float(mean_abs_delta / (mean_abs_x + 1e-12))
        dbg["active"] = 1
        state["debug_rows"].append(dbg)

        selected_new = _unflatten_spatial(flat_new.to(dtype=selected.dtype, device=selected.device), meta)
        out[sl] = selected_new
        return _pack_tensor(out, is_tuple)

    # 暴露调试信息，供 exp54 写出每步诊断 CSV
    hook.debug_rows = state["debug_rows"]  # type: ignore[attr-defined]
    return hook
