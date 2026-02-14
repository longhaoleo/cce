"""实验 2.4：特征正交性检验（Orthogonality Check）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp24_orthogonality(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp24",
        title="特征正交性检验",
        output_dir=output_dir,
        notes="""
1. 选取高频激活特征（如 Top-50）。
2. 计算解码方向向量两两余弦相似度矩阵。
3. 输出热力图并统计非对角平均相似度。
4. 验证近似正交性与语义解耦程度。
""",
    )
