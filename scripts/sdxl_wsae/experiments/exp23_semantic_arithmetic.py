"""实验 2.3：语义算术与特征交换（Semantic Arithmetic）。"""

from __future__ import annotations

from .placeholder import run_placeholder_experiment


def run_exp23_semantic_arithmetic(output_dir: str) -> None:
    run_placeholder_experiment(
        exp_id="exp23",
        title="语义算术与特征交换",
        output_dir=output_dir,
        notes="""
1. 采集 King/Man/Woman 的特征轨迹编码。
2. 在 SAE 空间执行 Enc(King)-Enc(Man)+Enc(Woman)。
3. 将合成特征轨迹解码后注入到目标 block，生成新图。
4. 评估是否逼近 Queen 语义。
""",
    )
