"""实验 2：背景与主体无损分离（Context Preservation）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp02_context_preservation(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp02",
        title="背景与主体无损分离",
        output_dir=output_dir,
        notes="""
1. 定位主体特征（如 dog）。
2. 进行特征擦除，生成 before/after 图像。
3. 计算像素差值图 |A-B|，并统计背景区域变化比例。
4. 与 ESD 或全模型微调结果做对比，验证局部可控性优势。
""",
    )
