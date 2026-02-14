"""实验 2.2：特征惯性/鲁棒性验证（Feature Inertia）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp22_feature_inertia(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp22",
        title="特征惯性（单步擦除 vs 多步擦除）",
        output_dir=output_dir,
        notes="""
1. 对同一语义特征设置两种擦除策略：
   - 单步擦除（例如 t=500）；
   - 多步擦除（例如 t=[600, 400]）。
2. 比较最终图像与中间特征曲线恢复情况。
3. 结论目标：验证扩散轨迹的积分鲁棒性。
""",
    )
