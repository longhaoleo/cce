"""
训练指标记录与曲线绘制。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class StepMetric:
    """单步训练指标。

    输入：
    - 各字段由训练循环在每个优化 step 填充。

    输出：
    - 可序列化记录对象，用于 jsonl 持久化与后续绘图。
    """

    ts_unix: float
    stage: str
    global_step: int
    local_step: int
    loss_total: float
    loss_recon: float
    loss_auxk: float
    loss_align: float
    loss_latent_decorr: float
    loss_auxk_term: float
    loss_align_term: float
    loss_latent_decorr_term: float
    align_weight: float
    time_branch_scale: float
    latent_active_frac: float
    dead_feature_frac: float
    lr_shared: float
    lr_adapter: float
    lr_time: float
    lr_spatial: float


class MetricsWriter:
    """训练指标写入器。

    输入：
    - output_root: 训练输出根目录。

    输出：
    - 提供 step/stage 指标写入能力。
    """

    def __init__(self, output_root: str):
        """初始化指标写入器并创建目录。

        输入：
        - output_root: 输出根目录。

        输出：
        - MetricsWriter 实例。
        """
        self.output_root = Path(output_root).expanduser().resolve()
        self.metrics_dir = self.output_root / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.step_metrics_path = self.metrics_dir / "step_metrics.jsonl"
        self.stage_metrics_path = self.metrics_dir / "stage_metrics.jsonl"

    def write_step(self, metric: StepMetric) -> None:
        """写入单步指标记录。

        输入：
        - metric: StepMetric 对象。

        输出：
        - 无；追加写入 `step_metrics.jsonl`。
        """
        with self.step_metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")

    def write_stage_summary(self, payload: Dict) -> None:
        """写入阶段汇总记录。

        输入：
        - payload: 阶段汇总字典。

        输出：
        - 无；追加写入 `stage_metrics.jsonl`。
        """
        record = dict(payload)
        record["ts_unix"] = float(time.time())
        with self.stage_metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    """读取 jsonl 文件。

    输入：
    - path: jsonl 文件路径。

    输出：
    - List[Dict]：逐行解析后的记录列表。
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return []
    out: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def plot_loss_curves(output_root: str) -> Dict[str, str]:
    """根据 step 指标绘制 loss 曲线图。

    输入：
    - output_root: 训练输出根目录。

    输出：
    - Dict[str,str]：返回生成图片路径；若环境缺少 matplotlib，则返回空字典。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    root = Path(output_root).expanduser().resolve()
    metrics_path = root / "metrics" / "step_metrics.jsonl"
    rows = read_jsonl(str(metrics_path))
    if not rows:
        return {}

    steps = [int(r["global_step"]) for r in rows]
    total = [float(r["loss_total"]) for r in rows]
    recon = [float(r["loss_recon"]) for r in rows]
    auxk = [float(r["loss_auxk"]) for r in rows]
    align = [float(r["loss_align"]) for r in rows]
    latent_decorr = [float(r.get("loss_latent_decorr", 0.0)) for r in rows]
    align_w = [float(r["align_weight"]) for r in rows]
    time_scale = [float(r.get("time_branch_scale", 1.0)) for r in rows]

    fig_dir = root / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 图1：四类损失曲线
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(steps, total, label="loss_total", linewidth=1.8)
    plt.plot(steps, recon, label="loss_recon", linewidth=1.4)
    plt.plot(steps, auxk, label="loss_auxk", linewidth=1.4)
    plt.plot(steps, align, label="loss_align", linewidth=1.4)
    plt.plot(steps, latent_decorr, label="loss_latent_decorr", linewidth=1.2)
    plt.xlabel("global_step")
    plt.ylabel("loss")
    plt.title("Shared SAE Training Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    loss_curve_path = fig_dir / "loss_curves.png"
    fig1.tight_layout()
    fig1.savefig(loss_curve_path, dpi=160)
    plt.close(fig1)

    # 图2：align_weight / time_branch_scale 调度曲线
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(steps, align_w, label="align_weight", color="#CC5500", linewidth=1.8)
    plt.plot(steps, time_scale, label="time_branch_scale", color="#3366CC", linewidth=1.8)
    plt.xlabel("global_step")
    plt.ylabel("weight / scale")
    plt.title("Training Schedules")
    plt.grid(alpha=0.25)
    plt.legend()
    align_curve_path = fig_dir / "align_weight_curve.png"
    fig2.tight_layout()
    fig2.savefig(align_curve_path, dpi=160)
    plt.close(fig2)

    return {
        "loss_curves": str(loss_curve_path),
        "align_weight_curve": str(align_curve_path),
    }
