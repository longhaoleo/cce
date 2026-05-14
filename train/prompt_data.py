"""
Prompt 数据读取与划分。
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass
class PromptRecord:
    """单条 prompt 记录。

    输入：
    - prompt_id: 全局编号（本模块生成）
    - prompt: 文本内容
    - seed: 采样随机种子

    输出：
    - PromptRecord 对象，可直接用于采样。
    """

    prompt_id: int
    prompt: str
    seed: int


def load_prompts_from_csv(csv_path: str, base_seed: int) -> List[PromptRecord]:
    """从 CSV 读取 prompt 列表。

    输入：
    - csv_path: CSV 文件路径，至少包含 `prompt` 列，若无则取首列。
    - base_seed: 当行内没有 seed 信息时，使用 `base_seed + row_idx`。

    输出：
    - List[PromptRecord]：过滤空行后的 prompt 记录列表。
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到 prompts_csv: {path}")

    out: List[PromptRecord] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头: {path}")
        fields = [str(x) for x in reader.fieldnames]
        prompt_col = "prompt" if "prompt" in fields else fields[0]
        seed_col = "evaluation_seed" if "evaluation_seed" in fields else ("seed" if "seed" in fields else "")

        for idx, row in enumerate(reader):
            prompt = str(row.get(prompt_col, "")).strip()
            if not prompt:
                continue
            if seed_col and str(row.get(seed_col, "")).strip():
                seed = int(row[seed_col])
            else:
                seed = int(base_seed) + int(idx)
            out.append(PromptRecord(prompt_id=len(out), prompt=prompt, seed=seed))
    return out


def split_prompt_records(
    records: Sequence[PromptRecord],
    *,
    split_seed: int,
    validation_prompts: int,
    stage2_train_prompts: int,
    calibration_prompts: int,
) -> Dict[str, List[PromptRecord]]:
    """按计划规则划分数据集。

    输入：
    - records: 全量 prompt 记录。
    - split_seed: 划分随机种子（保证可复现）。
    - validation_prompts: 验证集数量，必须与训练严格不重叠。
    - stage2_train_prompts: Stage2/3 训练集大小。
    - calibration_prompts: 校准集大小，可与训练集重叠。

    输出：
    - Dict[str, List[PromptRecord]]：包含 `validation/stage2/calibration/train_pool`。
    """
    if len(records) < int(validation_prompts) + int(stage2_train_prompts):
        raise ValueError("prompt 总量不足以切分 validation + stage2_train")

    rng = random.Random(int(split_seed))
    shuffled = list(records)
    rng.shuffle(shuffled)

    validation = shuffled[: int(validation_prompts)]
    train_pool = shuffled[int(validation_prompts) :]

    stage2 = train_pool[: int(stage2_train_prompts)]
    calibration = train_pool[: int(calibration_prompts)]

    return {
        "validation": list(validation),
        "train_pool": list(train_pool),
        "stage2": list(stage2),
        "calibration": list(calibration),
    }


def maybe_truncate(records: Sequence[PromptRecord], max_prompts_debug: int) -> List[PromptRecord]:
    """按调试参数裁剪 prompt 数量。

    输入：
    - records: 原始记录列表。
    - max_prompts_debug: 若 >0，只保留前 N 条。

    输出：
    - List[PromptRecord]：裁剪后的列表。
    """
    if int(max_prompts_debug) <= 0:
        return list(records)
    return list(records)[: int(max_prompts_debug)]


def iter_prompt_shards(records: Sequence[PromptRecord], shard_size: int) -> Iterator[List[PromptRecord]]:
    """按固定分片大小迭代 prompt 列表。

    输入：
    - records: 待切分记录。
    - shard_size: 每个 shard 包含的 prompt 数。

    输出：
    - Iterator[List[PromptRecord]]：逐片返回 prompt 子列表。
    """
    if int(shard_size) <= 0:
        raise ValueError("shard_size 必须 > 0")
    n = len(records)
    i = 0
    while i < n:
        j = min(i + int(shard_size), n)
        yield list(records[i:j])
        i = j


def summarize_split(split: Dict[str, Sequence[PromptRecord]]) -> Dict[str, int]:
    """统计数据划分大小，便于日志输出。

    输入：
    - split: 由 `split_prompt_records` 返回的字典。

    输出：
    - Dict[str, int]：每个切分名称对应的样本数量。
    """
    return {k: len(v) for k, v in split.items()}
