"""
vslz_wosae.py

用途：
- 基于 SDXL（Stable Diffusion XL）生成图像的同时，通过 attention_map_diffusers 的 hook
  收集各层注意力图（attention maps）。
- 对提示词（prompt）中两个概念（concept A / concept B）的注意力随扩散时间步的变化进行聚合，
  并计算二值 IoU 与连续重叠度等指标，最后可视化曲线。

依赖：
- diffusers（SDXL 管线）
- attention_map_diffusers（收集注意力图的工具）
- torch / numpy / matplotlib / pillow
"""

import os
import re
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from typing import List, Dict, Optional, Tuple

# 尝试导入必要依赖（如果缺失则提示安装并退出）
try:
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
    from attention_map_diffusers import attn_maps, init_pipeline
except ImportError as e:
    print(f"导入依赖失败：{e}")
    print("请安装依赖：pip install diffusers attention_map_diffusers torch matplotlib")
    exit(1)

# --- 辅助函数（从 Notebook 迁移/整理）---

def set_seed(seed: int) -> None:
    """固定随机种子，便于复现实验结果。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    """确保目录存在（不存在则创建）。"""
    os.makedirs(path, exist_ok=True)

def normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """把张量线性归一化到 [0, 1]。"""
    x = x - x.min()
    x = x / (x.max() + eps)
    return x

def _normalize_text_for_match(s: str) -> str:
    """为 token 匹配做文本规范化：去首尾空白、合并多空白为单空格。"""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _safe_filename(name: str) -> str:
    """把任意字符串转成相对安全的文件名片段（用于输出文件）。"""
    name = name.strip()
    name = re.sub(r"[^\w\.\-]+", "_", name, flags=re.UNICODE)
    return name[:180] if len(name) > 180 else name

def _matches_any_regex(text: str, patterns: Optional[List[str]]) -> bool:
    """text 是否匹配 patterns 中任意一个正则；patterns=None/空表示全匹配。"""
    if not patterns:
        return True
    return any(re.search(p, text) for p in patterns)

def configure_matplotlib_fonts(prefer_chinese: bool = True) -> bool:
    """
    配置 matplotlib 字体，尽量兼容中文。

    返回：
    - True：检测到并设置了较可能支持中文的字体
    - False：未检测到常见中文字体（图里的中文可能显示为方块/乱码）
    """
    matplotlib.rcParams["axes.unicode_minus"] = False

    if not prefer_chinese:
        return False

    preferred = [
        # Linux 常见
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Noto Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        # Windows 常见
        "Microsoft YaHei",
        "SimHei",
        # macOS 常见
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return True

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False

def _percentile_vmin_vmax(x: np.ndarray, p_low: float = 60.0, p_high: float = 99.5) -> Tuple[float, float]:
    """
    用分位数设置 vmin/vmax，让颜色对比更明显（避免大面积低值把热力图“压暗”）。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(x, p_low))
    vmax = float(np.percentile(x, p_high))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin + 1e-6:
        # 兜底：避免 vmax==vmin 导致全黑
        vmin = float(x.min())
        vmax = float(x.max() + 1e-6)
    return vmin, vmax

def _overlay_rgba_from_map(
    attn: np.ndarray,
    *,
    cmap_name: str = "turbo",
    alpha_max: float = 0.85,
    gamma: float = 1.6,
    p_low: float = 60.0,
    p_high: float = 99.5,
) -> np.ndarray:
    """
    把注意力图转成 RGBA 叠加图：
    - 颜色用 colormap
    - 透明度用 attn 强度（高值更不透明），让“重要区域”更醒目
    """
    attn = np.asarray(attn, dtype=np.float32)
    vmin, vmax = _percentile_vmin_vmax(attn, p_low=p_low, p_high=p_high)
    norm = np.clip((attn - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    # Matplotlib 3.7+：cm.get_cmap 已弃用，使用 matplotlib.colormaps
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H,W,4) float in [0,1]
    rgba[..., 3] = (norm ** gamma) * float(alpha_max)
    return rgba, vmin, vmax

def _maybe_add_colorbar(ax, *, cmap_name: str, vmin: float, vmax: float) -> None:
    """在 vmin/vmax 合法时添加 colorbar；否则跳过，避免 matplotlib 里除零/越界。"""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return
    if vmax <= vmin + 1e-6:
        return
    try:
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=matplotlib.colormaps.get_cmap(cmap_name))
    except Exception:
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cm.get_cmap(cmap_name))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

def _infer_attn_kind(layer_name: str) -> str:
    """
    基于层名推断注意力类型：
    - SD/SDXL UNet 里常见命名：attn1≈self-attn，attn2≈cross-attn（与文本 token 对齐）
    """
    ln = layer_name.lower()
    if "attn2" in ln or "cross" in ln:
        return "cross"
    if "attn1" in ln or "self" in ln:
        return "self"
    return "unknown"

def _infer_block_kind(layer_name: str) -> str:
    """基于层名粗略推断属于 down/mid/up/other 哪个块。"""
    ln = layer_name.lower()
    if "down" in ln:
        return "down"
    if "mid" in ln:
        return "mid"
    if "up" in ln:
        return "up"
    return "other"

def _group_layer_name(layer_name: str, group_mode: str) -> str:
    """
    把 layer_name 映射到分组名，便于“按块/按注意力类型”看随时间变化。

    group_mode:
    - "none": 全部合到 "all"
    - "block": down/mid/up/other
    - "block_attn": down.cross / down.self / ...
    - "layer": 每个 layer 单独一组（文件会很多）
    """
    group_mode = (group_mode or "block_attn").lower()
    if group_mode == "none":
        return "all"
    if group_mode == "layer":
        return layer_name
    block = _infer_block_kind(layer_name)
    if group_mode == "block":
        return block
    attn = _infer_attn_kind(layer_name)
    return f"{block}.{attn}"

def find_token_indices_by_subsequence(tokenizer, prompt: str, phrase: str, debug: bool = False) -> List[int]:
    """
    在 prompt 的 token 序列里，查找 phrase 对应的连续子序列位置（返回 token 索引列表）。

    说明：
    - 这里用 tokenizer 得到 prompt_ids 与 phrase_ids，然后做“连续子序列”匹配。
    - 返回的索引是 phrase 在 prompt token 序列里的起止范围（包含端点）。
    """
    prompt_norm = _normalize_text_for_match(prompt)
    phrase_norm = _normalize_text_for_match(phrase)

    prompt_ids = tokenizer(prompt_norm, add_special_tokens=True)["input_ids"]
    phrase_ids = tokenizer(phrase_norm, add_special_tokens=False)["input_ids"]

    if debug:
        print(f"Prompt tokens: {prompt_ids}")
        print(f"Phrase tokens: {phrase_ids}")

    if len(phrase_ids) == 0:
        return []

    # 连续子序列匹配：在 prompt_ids 中寻找一段与 phrase_ids 完全相同的连续片段
    for i in range(len(prompt_ids) - len(phrase_ids) + 1):
        if prompt_ids[i:i+len(phrase_ids)] == phrase_ids:
            return list(range(i, i+len(phrase_ids)))
    
    return []

def _is_perfect_square(n: int) -> bool:
    if n <= 0:
        return False
    r = int(math.isqrt(int(n)))
    return r * r == int(n)

def _attn_to_bthw(a: torch.Tensor, *, layer_name: str = "") -> Optional[torch.Tensor]:
    """
    尝试把 attention_map_diffusers 的张量形状规范化为 (B, T, H, W)。

    兼容常见形状（不同版本/实现可能不同）：
    - (B, Heads, H, W, T) / (B, Heads, T, H, W) -> 先 sum(heads)
    - (B, H, W, T) -> permute 到 (B, T, H, W)
    - (B, T, H, W) -> 原样返回
    - (B, Heads, HW, T) / (B, Heads, T, HW) -> sum(heads) 后 reshape
    - (B, HW, T) / (B, T, HW) -> reshape 为 (B, T, H, W)，其中 H=W=sqrt(HW)
    """
    if not isinstance(a, torch.Tensor):
        return None

    # 先处理明显含 heads 的 5D 情况
    if a.dim() == 5:
        a = a.sum(1)  # sum heads

    # 有些实现会给 4D：(B, Heads, HW, T) 或 (B, Heads, T, HW)
    if a.dim() == 4:
        # 启发式：第二维像 heads（<=32），且后两维不同时可能是 HW/T
        if a.shape[1] <= 32 and a.shape[2] != a.shape[3]:
            a = a.sum(1)  # -> (B, *, *)

    # 3D：把 HW 还原成 H,W
    if a.dim() == 3:
        b, d1, d2 = a.shape
        # 两种常见： (B, HW, T) 或 (B, T, HW)
        if _is_perfect_square(d1) and not _is_perfect_square(d2):
            hw, t = d1, d2
            h = w = int(math.isqrt(int(hw)))
            a = a.reshape(b, h, w, t).permute(0, 3, 1, 2)  # (B,T,H,W)
            return a
        if _is_perfect_square(d2) and not _is_perfect_square(d1):
            t, hw = d1, d2
            h = w = int(math.isqrt(int(hw)))
            a = a.reshape(b, t, h, w)  # (B,T,H,W)
            return a
        if _is_perfect_square(d1) and _is_perfect_square(d2):
            # 都是平方数时，通常 HW 更大；用更大的当 HW
            if d1 >= d2:
                hw, t = d1, d2
                h = w = int(math.isqrt(int(hw)))
                a = a.reshape(b, h, w, t).permute(0, 3, 1, 2)
                return a
            else:
                t, hw = d1, d2
                h = w = int(math.isqrt(int(hw)))
                a = a.reshape(b, t, h, w)
                return a
        # 无法推断 HW->H,W
        return None

    # 4D：要么 (B,T,H,W)，要么 (B,H,W,T)
    if a.dim() == 4:
        b, d1, d2, d3 = a.shape
        # (B,T,H,W)：后两维通常是方形空间分辨率（8/16/32/64...）
        if d2 == d3 and d2 <= 256:
            return a
        # (B,H,W,T)：前两维是空间分辨率
        if d1 == d2 and d1 <= 256:
            return a.permute(0, 3, 1, 2)

        # 兜底：如果最后一维看起来更像 tokens（通常 > 空间分辨率），也尝试当作 (B,H,W,T)
        if d1 <= 256 and d2 <= 256 and d3 > max(d1, d2):
            return a.permute(0, 3, 1, 2)

        return None

    return None

def aggregate_attention_for_timestep(attn_maps_at_step: Dict, keep_conditional: bool = True):
    """
    聚合单个时间步（timestep）的注意力图：跨层（layers）/跨头（heads）求和再求平均。

    期望输出形状：
    - (batch, tokens, H, W)

    备注：
    - attention_map_diffusers 的输出 shape 在不同实现/版本可能略有差异（例如 tokens 维度位置不同）。
      这里用了一些“形状启发式”去对齐到 (B, T, H, W)。
    - keep_conditional=True 时，会在 CFG（classifier-free guidance）场景下仅保留 conditional 分支。
    """
    token_maps_sum = None
    token_maps_count = 0
    
    # 遍历该 timestep 下的所有层（layer），把每层的注意力图累加起来
    for layer_name, a in attn_maps_at_step.items():
        # 规范化为 (B,T,H,W)
        a = _attn_to_bthw(a, layer_name=layer_name)
        if a is None:
            continue
        
        # CFG（classifier-free guidance）场景：batch 维可能把 uncond / cond 拼在一起
        if keep_conditional and a.shape[0] == 2:
            a = a.chunk(2)[1] # 只取 conditional 部分
        elif keep_conditional and a.shape[0] > 2 and a.shape[0] % 2 == 0:
            a = a.chunk(2, dim=0)[1]

        # 累加：不同层的注意力图空间分辨率可能不同，必要时插值到同一分辨率
        if token_maps_sum is None:
            token_maps_sum = a.clone()
        else:
            # 少数情况下不同层的 token 维长度不一致：对齐到最小长度（避免直接报错）
            if a.shape[1] != token_maps_sum.shape[1]:
                min_len = min(int(a.shape[1]), int(token_maps_sum.shape[1]))
                a = a[:, :min_len]
                token_maps_sum = token_maps_sum[:, :min_len]
            if a.shape[-2:] != token_maps_sum.shape[-2:]:
                a = F.interpolate(a, size=token_maps_sum.shape[-2:], mode="bilinear", align_corners=False)
            token_maps_sum += a
        token_maps_count += 1
    
    if token_maps_count == 0:
        return None
        
    # 求层平均（对层的简单平均；heads 在前面已求和）
    return token_maps_sum / token_maps_count

def compute_iou(map1, map2, thr=0.3):
    """把注意力图二值化后计算 IoU（Intersection over Union）。"""
    map1_bin = (map1 >= thr).float()
    map2_bin = (map2 >= thr).float()
    intersection = (map1_bin * map2_bin).sum()
    union = torch.clamp(map1_bin + map2_bin, 0, 1).sum()
    return (intersection / (union + 1e-8)).item()

def compute_overlap_metrics(map1, map2):
    """连续重叠度：sum(min(a,b)) / sum(max(a,b))。"""
    intersection = torch.minimum(map1, map2).sum()
    union = torch.maximum(map1, map2).sum()
    return (intersection / (union + 1e-8)).item()

def save_concept_maps_figure(
    timestep: int,
    step_idx: int,
    group_name: str,
    concept_a: str,
    concept_b: str,
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    iou: float,
    overlap: float,
    out_path: str,
    *,
    base_image: Optional[Image.Image] = None,
    overlay_on_image: bool = True,
    overlay_cmap_a: str = "turbo",
    overlay_cmap_b: str = "turbo",
    overlay_cmap_overlap: str = "viridis",
    overlay_alpha_max: float = 0.85,
    overlay_gamma: float = 1.6,
    overlay_p_low: float = 60.0,
    overlay_p_high: float = 99.5,
    contour_quantile: float = 0.92,
    chinese_labels: bool = True,
) -> None:
    """
    保存单个 timestep 的可视化：
    - 概念 A 注意力热力图
    - 概念 B 注意力热力图
    - 连续重叠度热力图（min(map_a, map_b)）
    """
    overlap_map = torch.minimum(map_a, map_b)

    plt.figure(figsize=(12, 4))
    if chinese_labels:
        title = (
            f"组: {group_name} | idx={step_idx} timestep={timestep}\n"
            f"{concept_a} vs {concept_b} | IoU={iou:.3f} Overlap={overlap:.3f}"
        )
    else:
        title = (
            f"Group: {group_name} | idx={step_idx} timestep={timestep}\n"
            f"{concept_a} vs {concept_b} | IoU={iou:.3f} Overlap={overlap:.3f}"
        )
    plt.suptitle(title, fontsize=11)

    plt.subplot(1, 3, 1)
    if base_image is not None and overlay_on_image:
        w, h = base_image.size
        base_np = np.asarray(base_image.convert("RGB"))
        plt.imshow(base_np)
        up = F.interpolate(
            map_a[None, None].detach().float().cpu(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        rgba, vmin, vmax = _overlay_rgba_from_map(
            up,
            cmap_name=overlay_cmap_a,
            alpha_max=overlay_alpha_max,
            gamma=overlay_gamma,
            p_low=overlay_p_low,
            p_high=overlay_p_high,
        )
        plt.imshow(rgba)
        # 用等高线强调“最重要”的区域
        thr = float(np.quantile(up, contour_quantile))
        plt.contour(up, levels=[thr], colors=["cyan"], linewidths=1.2)
        plt.title(f"A: {concept_a} (overlay)")
    else:
        a_np = map_a.detach().float().cpu().numpy()
        vmin, vmax = _percentile_vmin_vmax(a_np, p_low=overlay_p_low, p_high=overlay_p_high)
        plt.imshow(a_np, cmap=overlay_cmap_a, vmin=vmin, vmax=vmax)
        plt.title(f"A: {concept_a}")
    plt.axis("off")
    _maybe_add_colorbar(plt.gca(), cmap_name=overlay_cmap_a, vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 2)
    if base_image is not None and overlay_on_image:
        w, h = base_image.size
        base_np = np.asarray(base_image.convert("RGB"))
        plt.imshow(base_np)
        up = F.interpolate(
            map_b[None, None].detach().float().cpu(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        rgba, vmin, vmax = _overlay_rgba_from_map(
            up,
            cmap_name=overlay_cmap_b,
            alpha_max=overlay_alpha_max,
            gamma=overlay_gamma,
            p_low=overlay_p_low,
            p_high=overlay_p_high,
        )
        plt.imshow(rgba)
        thr = float(np.quantile(up, contour_quantile))
        plt.contour(up, levels=[thr], colors=["cyan"], linewidths=1.2)
        plt.title(f"B: {concept_b} (overlay)")
    else:
        b_np = map_b.detach().float().cpu().numpy()
        vmin, vmax = _percentile_vmin_vmax(b_np, p_low=overlay_p_low, p_high=overlay_p_high)
        plt.imshow(b_np, cmap=overlay_cmap_b, vmin=vmin, vmax=vmax)
        plt.title(f"B: {concept_b}")
    plt.axis("off")
    _maybe_add_colorbar(plt.gca(), cmap_name=overlay_cmap_b, vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 3)
    if base_image is not None and overlay_on_image:
        w, h = base_image.size
        base_np = np.asarray(base_image.convert("RGB"))
        plt.imshow(base_np)
        up = F.interpolate(
            overlap_map[None, None].detach().float().cpu(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        rgba, vmin, vmax = _overlay_rgba_from_map(
            up,
            cmap_name=overlay_cmap_overlap,
            alpha_max=overlay_alpha_max,
            gamma=overlay_gamma,
            p_low=overlay_p_low,
            p_high=overlay_p_high,
        )
        plt.imshow(rgba)
        thr = float(np.quantile(up, contour_quantile))
        plt.contour(up, levels=[thr], colors=["magenta"], linewidths=1.2)
        plt.title("Overlap (overlay)")
    else:
        o_np = overlap_map.detach().float().cpu().numpy()
        vmin, vmax = _percentile_vmin_vmax(o_np, p_low=overlay_p_low, p_high=overlay_p_high)
        plt.imshow(o_np, cmap=overlay_cmap_overlap, vmin=vmin, vmax=vmax)
        plt.title("Overlap: min(A,B)")
    plt.axis("off")
    _maybe_add_colorbar(plt.gca(), cmap_name=overlay_cmap_overlap, vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def write_metrics_csv(out_path: str, rows: List[Dict]) -> None:
    """把每个 timestep 的指标保存到 CSV，方便后续做更细的分析。"""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def write_top_events_csv(
    out_path: str,
    rows: List[Dict],
    *,
    metric_key: str = "overlap",
    top_k: int = 5,
) -> None:
    """按 metric_key（默认 overlap）挑选 top-k timestep，保存为 CSV。"""
    if not rows:
        return
    metric_key = metric_key or "overlap"
    top_k = max(int(top_k), 0)
    if top_k == 0:
        return

    sorted_rows = sorted(rows, key=lambda r: float(r.get(metric_key, 0.0)), reverse=True)[:top_k]
    fieldnames = list(sorted_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted_rows)

def load_diffusers_pipeline(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_safetensors: bool = True,
    variant: Optional[str] = "fp16",
) -> DiffusionPipeline:
    """
    加载 diffusers pipeline，兼容 SD1.5 与 SDXL。

    说明：
    - 统一使用 DiffusionPipeline.from_pretrained，并在必要时回退参数（例如某些模型没有 fp16 variant）。
    - 返回的 pipe 可能是 StableDiffusionPipeline / StableDiffusionXLPipeline 或其它 text2image pipeline。
    """
    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": use_safetensors,
    }
    if variant:
        kwargs["variant"] = variant

    def _try_load(extra_kwargs: Optional[Dict] = None) -> DiffusionPipeline:
        load_kwargs = dict(kwargs)
        if extra_kwargs:
            load_kwargs.update(extra_kwargs)
        return DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

    disable_safety_and_extractor = {
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }

    # 多策略加载：优先“最全参数”，失败后逐步回退（去掉 variant、禁用 safety/feature_extractor）
    attempt_kwargs: List[Dict] = []
    attempt_kwargs.append(dict(kwargs))
    if "variant" in kwargs:
        k = dict(kwargs)
        k.pop("variant", None)
        attempt_kwargs.append(k)
    attempt_kwargs.append({**dict(kwargs), **disable_safety_and_extractor})
    if "variant" in kwargs:
        k = dict(kwargs)
        k.pop("variant", None)
        attempt_kwargs.append({**k, **disable_safety_and_extractor})

    last_exc: Optional[BaseException] = None
    pipe = None
    for ak in attempt_kwargs:
        try:
            pipe = DiffusionPipeline.from_pretrained(model_id, **ak)
            last_exc = None
            break
        except TypeError as e:
            # 某些版本/模型不接受 use_safetensors/variant 等参数：最小化参数再试一次
            last_exc = e
            try:
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
                last_exc = None
                break
            except Exception as e2:
                last_exc = e2
                continue
        except ValueError as e:
            # 指定 variant 但不存在：继续尝试后续（variantless）策略
            last_exc = e
            continue
        except OSError as e:
            # 本地目录缺 safety_checker / feature_extractor 配置或权重：继续尝试禁用策略
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            continue

    if pipe is None:
        assert last_exc is not None
        raise last_exc

    pipe = pipe.to(device)

    # 这里做一个“类型提示”，方便用户确认加载的到底是 SD1.5 还是 SDXL
    if isinstance(pipe, StableDiffusionXLPipeline):
        print("已加载管线类型：StableDiffusionXLPipeline（SDXL）")
    elif isinstance(pipe, StableDiffusionPipeline):
        print("已加载管线类型：StableDiffusionPipeline（SD 1.x/2.x）")
    else:
        print(f"已加载管线类型：{pipe.__class__.__name__}")

    return pipe


# --- 主类：生成 + 统计 + 可视化 ---

class ConceptInteractionVisualizer:
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", dtype=torch.float16):
        # 兼容无 CUDA 环境：自动回退到 CPU，并避免在 CPU 上用 float16（多数算子不支持/很慢）
        if device == "cuda" and not torch.cuda.is_available():
            print("检测到 CUDA 不可用，自动切换到 CPU（dtype 将改为 float32）。")
            device = "cpu"
            dtype = torch.float32
        elif device == "cpu" and dtype == torch.float16:
            dtype = torch.float32

        self.device = device
        self.mpl_chinese_ok = configure_matplotlib_fonts(prefer_chinese=True)
        if not self.mpl_chinese_ok:
            print("提示：未检测到常见中文字体，图里的中文可能显示异常；将自动用英文标题。")
        print(f"使用设备：{device}，数据类型：{dtype}")
        # 加载模型（兼容 SDXL / SD1.5）
        print(f"正在加载模型：{model_id}...")
        # 注意：很多 SD1.5 仓库没有 "fp16" variant；这里会自动回退
        self.pipe = load_diffusers_pipeline(model_id, device=device, dtype=dtype, use_safetensors=True)
        
        # 初始化 hook：使得生成过程中的注意力图会被收集到全局的 attn_maps 中
        self.pipe = init_pipeline(self.pipe)
        print("模型加载完成，hooks 已初始化。")

    def run(
        self,
        prompt: str,
        concept_a: str,
        concept_b: str,
        steps: int = 25,
        seed: int = 42,
        output_dir: str = "output",
        *,
        layer_regex: Optional[List[str]] = None,
        group_mode: str = "block_attn",
        attn_kind: str = "cross",
        print_layer_names: bool = True,
        save_per_timestep_maps: bool = True,
        timestep_stride: int = 1,
        max_saved_timesteps_per_group: int = 60,
        iou_threshold: float = 0.4,
        overlay_maps_on_image: bool = True,
        interaction_top_k: int = 5,
        interaction_quantile: float = 0.9,
    ):
        ensure_dir(output_dir)
        set_seed(seed)
        
        # 1) 清空上一轮的注意力图缓存（避免跨运行污染）
        attn_maps.clear()
        
        # 2) 运行生成：在扩散过程中 hook 会不断往 attn_maps 里写入每个 timestep 的注意力图
        print(f"正在生成图像，prompt：'{prompt}'...")
        image = self.pipe(prompt, num_inference_steps=steps).images[0]
        image.save(os.path.join(output_dir, "generated_image.png"))
        
        # 3) 按时间步分析两个概念的注意力交互（IoU / Overlap / 强度）
        print("正在按时间步分析注意力图...")
        
        # 找出 concept_a / concept_b 在 prompt 里的 token 索引
        # - SD1.5 通常只有 tokenizer
        # - SDXL 有 tokenizer 与 tokenizer_2（两个 text encoder）
        if not hasattr(self.pipe, "tokenizer") or self.pipe.tokenizer is None:
            print("错误：当前 pipeline 没有 tokenizer，无法按 token 计算概念激活。")
            return

        ids_a = find_token_indices_by_subsequence(self.pipe.tokenizer, prompt, concept_a)
        ids_b = find_token_indices_by_subsequence(self.pipe.tokenizer, prompt, concept_b)

        # 如果没找到，且存在 tokenizer_2，则再尝试 tokenizer_2（主要用于 SDXL）
        tokenizer_2 = getattr(self.pipe, "tokenizer_2", None)
        if tokenizer_2 is not None:
            if not ids_a:
                ids_a = find_token_indices_by_subsequence(tokenizer_2, prompt, concept_a)
            if not ids_b:
                ids_b = find_token_indices_by_subsequence(tokenizer_2, prompt, concept_b)
            
        print(f"Token Indices - {concept_a}: {ids_a}, {concept_b}: {ids_b}")
        
        if not ids_a or not ids_b:
            print("错误：未能在 prompt 中定位一个或两个概念的 token。请检查拼写/空格。")
            return

        # 注意：扩散 timestep 往往是“高噪声 -> 低噪声”，这里按从大到小排序（更符合生成进度）
        timesteps = sorted([t for t in attn_maps.keys()], reverse=True)
        if not timesteps:
            print("错误：attn_maps 为空（没有采集到注意力图）。请确认 hooks 是否生效。")
            return
        print(f"本次共采集到 {len(timesteps)} 个 timestep（用于随时间分析）。")
        if len(timesteps) <= 12:
            print("timestep 列表：", [int(x) for x in timesteps])
        else:
            head = [int(x) for x in timesteps[:5]]
            tail = [int(x) for x in timesteps[-5:]]
            print("timestep 预览（前 5 / 后 5）：", head, "...", tail)

        # 取所有 timestep 的 layer 名并集，避免“某些 timestep 才出现的层”（例如 mid_block）被漏掉
        all_layers_set = set()
        for t in timesteps:
            all_layers_set.update(attn_maps[t].keys())
        example_layers = sorted(list(all_layers_set))
        if print_layer_names:
            print("采集到的 layer 名（示例，可能很多）：")
            for ln in example_layers[:120]:
                print(" -", ln)
            if len(example_layers) > 120:
                print(f" - ...（共 {len(example_layers)} 个）")

        # 默认只看 cross-attn（与文本 token 对齐），否则“按 token 索引取概念”可能不成立
        attn_kind = (attn_kind or "cross").lower()
        selected_layers = []
        for ln in example_layers:
            if not _matches_any_regex(ln, layer_regex):
                continue
            kind = _infer_attn_kind(ln)
            if attn_kind in ("any", "all"):
                selected_layers.append(ln)
            elif kind == attn_kind:
                selected_layers.append(ln)

        if not selected_layers:
            print("警告：按 layer_regex/attn_kind 过滤后没有任何层可用，将回退为全层聚合。")
            selected_layers = example_layers

        # 分组：group_name -> [layer_name, ...]
        groups: Dict[str, List[str]] = {}
        for ln in selected_layers:
            g = _group_layer_name(ln, group_mode=group_mode)
            groups.setdefault(g, []).append(ln)
        print("分组统计（组名: 层数）：")
        for g in sorted(groups.keys()):
            print(f" - {g}: {len(groups[g])}")

        # 每个 group 各自存一份时间序列指标
        group_series: Dict[str, Dict[str, List[float]]] = {}
        group_csv_rows: Dict[str, List[Dict]] = {}
        for g in groups.keys():
            group_series[g] = {"ious": [], "overlaps": [], "act_a": [], "act_b": []}
            group_csv_rows[g] = []

        # 输出目录：按组保存
        blocks_dir = os.path.join(output_dir, "blocks")
        ensure_dir(blocks_dir)
        
        # 遍历每个 timestep：对每个 group 聚合其层 -> 提取概念 A/B -> 计算指标
        saved_counts_per_group: Dict[str, int] = {g: 0 for g in groups.keys()}
        timestep_stride = max(int(timestep_stride), 1)
        max_saved_timesteps_per_group = max(int(max_saved_timesteps_per_group), 0)
        for step_idx, t in enumerate(timesteps):
            step_layers = attn_maps[t]
            for g, layer_list in groups.items():
                sub = {ln: step_layers[ln] for ln in layer_list if ln in step_layers}
                agg_map = aggregate_attention_for_timestep(sub)  # (B, Tokens, H, W)（期望 cross-attn）
                if agg_map is None:
                    continue

                # (1, Tokens, H, W) -> (Tokens, H, W)
                agg_map = agg_map[0]

                # 概念 A/B：对 token 取均值 -> 归一化
                map_a = normalize_01(agg_map[ids_a].mean(dim=0))
                map_b = normalize_01(agg_map[ids_b].mean(dim=0))

                # 指标
                iou = compute_iou(map_a, map_b, thr=float(iou_threshold))
                overlap = compute_overlap_metrics(map_a, map_b)
                act_a = map_a.mean().item()
                act_b = map_b.mean().item()

                group_series[g]["ious"].append(iou)
                group_series[g]["overlaps"].append(overlap)
                group_series[g]["act_a"].append(act_a)
                group_series[g]["act_b"].append(act_b)

                group_csv_rows[g].append(
                    {
                        "step_index": step_idx,
                        "timestep": int(t),
                        "iou": float(iou),
                        "overlap": float(overlap),
                        "act_a_mean": float(act_a),
                        "act_b_mean": float(act_b),
                    }
                )

                # 可选：保存逐 timestep 的热力图（注意：组多/步多会产出大量图片）
                if (
                    save_per_timestep_maps
                    and (step_idx % timestep_stride == 0)
                    and (max_saved_timesteps_per_group == 0 or saved_counts_per_group[g] < max_saved_timesteps_per_group)
                ):
                    g_dir = os.path.join(blocks_dir, _safe_filename(g))
                    ensure_dir(g_dir)
                    out_path = os.path.join(g_dir, f"timestep_{step_idx:04d}_t{int(t)}.png")
                    save_concept_maps_figure(
                        timestep=int(t),
                        step_idx=step_idx,
                        group_name=g,
                        concept_a=concept_a,
                        concept_b=concept_b,
                        map_a=map_a,
                        map_b=map_b,
                        iou=float(iou),
                        overlap=float(overlap),
                        out_path=out_path,
                        base_image=image,
                        overlay_on_image=bool(overlay_maps_on_image),
                        chinese_labels=self.mpl_chinese_ok,
                    )
                    saved_counts_per_group[g] += 1

        # 4) 输出：每组一张曲线 + 每组一份 CSV
        for g in groups.keys():
            g_dir = os.path.join(blocks_dir, _safe_filename(g))
            ensure_dir(g_dir)

            series = group_series[g]
            # 若该组完全没数据（例如层缺失），跳过
            if len(series["ious"]) == 0:
                continue

            self.plot_metrics(
                timesteps=timesteps,
                ious=series["ious"],
                overlaps=series["overlaps"],
                act_a=series["act_a"],
                act_b=series["act_b"],
                out_dir=g_dir,
                c_a=concept_a,
                c_b=concept_b,
                title_prefix=(f"组：{g}" if self.mpl_chinese_ok else f"Group: {g}"),
                interaction_quantile=float(interaction_quantile),
            )
            write_metrics_csv(os.path.join(g_dir, "metrics.csv"), group_csv_rows[g])
            write_top_events_csv(
                os.path.join(g_dir, "top_interaction.csv"),
                group_csv_rows[g],
                metric_key="overlap",
                top_k=int(interaction_top_k),
            )

        print(f"分析完成，结果已保存到：{output_dir}")

    def plot_metrics(
        self,
        timesteps,
        ious,
        overlaps,
        act_a,
        act_b,
        out_dir,
        c_a,
        c_b,
        title_prefix: str = "",
        interaction_quantile: float = 0.9,
    ):
        # x 轴：这里用“索引”作为生成进度（从 0 到 N-1）
        # 也可以改成直接画 timestep 值，但不同 scheduler 的 timestep 定义可能不同
        x = range(len(timesteps)) 
        
        plt.figure(figsize=(12, 5))
        
        # 图 1：交互强度（IoU / Overlap）
        plt.subplot(1, 2, 1)
        plt.plot(x, ious, label='IoU (Binary)', marker='o', color="#d62728")      # red
        plt.plot(x, overlaps, label='Overlap (Continuous)', marker='s', color="#1f77b4")  # blue
        title_1 = f"{c_a} vs {c_b}"
        if title_prefix:
            title_1 = f"{title_prefix}\n{title_1}"
        plt.title(title_1)
        plt.xlabel("Generation Progress (Step Index)")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # “什么时候交互最强”：按 overlap 的分位数阈值高亮 + 标注 top-3
        if len(overlaps) > 0:
            q = float(interaction_quantile)
            q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
            thr = float(np.quantile(np.asarray(overlaps, dtype=np.float32), q))
            mask = np.asarray(overlaps, dtype=np.float32) >= thr
            if mask.any():
                plt.fill_between(
                    x,
                    0,
                    1,
                    where=mask,
                    color="#1f77b4",
                    alpha=0.10,
                    transform=plt.gca().get_xaxis_transform(),
                    label="_nolegend_",
                )
                plt.axhline(thr, color="#1f77b4", alpha=0.35, linestyle="--", linewidth=1)

            top_k = min(3, len(overlaps))
            top_idx = sorted(range(len(overlaps)), key=lambda i: overlaps[i], reverse=True)[:top_k]
            for i in top_idx:
                plt.scatter([i], [overlaps[i]], color="black", s=35, zorder=5)
                plt.annotate(f"{i}", (i, overlaps[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        
        # 图 2：各自概念的平均注意力强度（归一化后）
        plt.subplot(1, 2, 2)
        plt.plot(x, act_a, label=f'{c_a} Intensity', color="#2ca02c")   # green
        plt.plot(x, act_b, label=f'{c_b} Intensity', color="#ff7f0e")   # orange
        plt.title("激活强度随时间变化" if self.mpl_chinese_ok else "Activation Intensity Over Time")
        plt.xlabel("Generation Progress (Step Index)")
        plt.ylabel("Mean Normalized Attention")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "temporal_interaction.png"))
        plt.close()

if __name__ == "__main__":
    # 示例用法：
    # - MODEL_ID 可以是 Hugging Face 上的模型 ID，也可以改成本地模型目录
    # MODEL_ID = os.path.expanduser("~/datasets/sd1.5/")
    MODEL_ID = os.path.expanduser("~/datasets/sd-xl/sdxl_diffusers_fp16/")
    
    visualizer = ConceptInteractionVisualizer(model_id=MODEL_ID)
    
    # PROMPT = "a child hold kitchen knife on the table, scary lighting."
    # CONCEPT_A = "child"
    # CONCEPT_B = "knife"
    PROMPT = "A red cat and a blue dog are playing together in a sunny park."
    CONCEPT_A = "red cat"
    CONCEPT_B = "blue dog"

    # 你关心“token A/B 的激活与交互”，通常应优先看 cross-attn（attn2），因为它与文本 token 对齐。
    # 输出结构：
    # - ./concept_vis_output/generated_image.png
    # - ./concept_vis_output/blocks/<组名>/temporal_interaction.png   （每组一张随时间曲线）
    # - ./concept_vis_output/blocks/<组名>/metrics.csv               （每组时间序列数值）
    # - ./concept_vis_output/blocks/<组名>/timestep_XXXX_tYYYY.png    （可选：逐 timestep 热力图）
    visualizer.run(
        PROMPT,
        CONCEPT_A,
        CONCEPT_B,
        steps=20,
        output_dir="./concept_vis_output",
        group_mode="layer",        # "none" | "block" | "block_attn" | "layer"
        attn_kind="any",              # "cross"(默认) | "self" | "unknown" | "any"
        layer_regex=None,               # 例如只看 up_blocks: [r"up"]
        save_per_timestep_maps=True,
        timestep_stride=1,              # 每隔 N 个 timestep 保存一张热力图
        max_saved_timesteps_per_group=30,  # 每组最多保存多少张（防止输出爆炸；0 表示不限制）
        iou_threshold=0.4,
        print_layer_names=True,         # 先打印 layer 名，便于你写 layer_regex 精确挑选“块”
    )
