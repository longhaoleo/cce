#!/usr/bin/env python3
"""Analyze whether compositional concept features are unions or new features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()


DEFAULT_BLOCKS = [
    "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    "unet.up_blocks.0.attentions.0",
    "unet.up_blocks.0.attentions.1",
]


def _safe_name(value: str) -> str:
    name = str(value).strip()
    name = re.sub(r"[^\w\.\-]+", "_", name, flags=re.UNICODE)
    return name[:180] if len(name) > 180 else name


def block_short_name(block: str) -> str:
    name = str(block).strip()
    m = re.search(r"up_blocks\.(\d+)\.attentions\.(\d+)", name)
    if m:
        return f"up.{m.group(1)}.{m.group(2)}"
    m = re.search(r"down_blocks\.(\d+)\.attentions\.(\d+)", name)
    if m:
        return f"down.{m.group(1)}.{m.group(2)}"
    m = re.search(r"mid_block\.attentions\.(\d+)", name)
    if m:
        return f"mid.{m.group(1)}"
    return _safe_name(name)


def _parse_pair(raw: str) -> Tuple[str, str, str]:
    parts = [x.strip() for x in str(raw).split(":")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"pair must be composite:atomic_a:atomic_b, got {raw!r}")
    return parts[0], parts[1], parts[2]


def _load_top_features(path: Path, *, top_k: int) -> List[int]:
    if not path.exists():
        return []
    rows: List[Tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"feature_id", "score"}.issubset(fields):
            raise ValueError(f"{path} missing feature_id/score columns")
        for row in reader:
            rows.append((int(row["feature_id"]), float(row["score"])))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [fid for fid, _score in rows[: int(top_k)]]


def _concept_dir(root: Path, block: str, concept: str) -> Path:
    return root / block_short_name(block) / str(concept)


def _score_row(
    *,
    root: Path,
    block: str,
    composite: str,
    atomic_a: str,
    atomic_b: str,
    top_k: int,
) -> Dict[str, object]:
    comp = _load_top_features(_concept_dir(root, block, composite) / "top_positive_features.csv", top_k=top_k)
    a = _load_top_features(_concept_dir(root, block, atomic_a) / "top_positive_features.csv", top_k=top_k)
    b = _load_top_features(_concept_dir(root, block, atomic_b) / "top_positive_features.csv", top_k=top_k)
    comp_set = set(comp)
    a_set = set(a)
    b_set = set(b)
    union_set = a_set | b_set
    denom = max(1, len(comp_set))
    return {
        "composite": composite,
        "atomic_a": atomic_a,
        "atomic_b": atomic_b,
        "block": block,
        "top_k": int(top_k),
        "n_composite": len(comp_set),
        "n_atomic_a": len(a_set),
        "n_atomic_b": len(b_set),
        "overlap_with_a": len(comp_set & a_set) / float(denom),
        "overlap_with_b": len(comp_set & b_set) / float(denom),
        "union_coverage": len(comp_set & union_set) / float(denom),
        "new_feature_ratio": len(comp_set - union_set) / float(denom),
        "atomic_overlap": len(a_set & b_set) / float(max(1, len(a_set | b_set))),
        "missing": int(not comp or not a or not b),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature overlap analysis for compositional concepts")
    ap.add_argument("--sae_root", type=str, default="sae_data/sae_x8_time_decorr03")
    ap.add_argument("--concept_root", type=str, default="", help="Override concept-dig root; default is <sae_root>/concept-dig")
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["dog_glasses:dog:glasses", "red_car:red:car", "flower_van_gogh:flower:van_gogh"],
        help="Triples formatted as composite:atomic_a:atomic_b",
    )
    ap.add_argument("--blocks", nargs="+", default=list(DEFAULT_BLOCKS))
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_csv", type=str, default="image_output/compositional_v0/feature_overlap.csv")
    args = ap.parse_args()

    root = Path(args.concept_root).expanduser() if str(args.concept_root).strip() else Path(args.sae_root).expanduser() / "concept-dig"
    root = root.resolve()
    rows: List[Dict[str, object]] = []
    for raw_pair in args.pairs:
        composite, atomic_a, atomic_b = _parse_pair(raw_pair)
        for block in args.blocks:
            rows.append(
                _score_row(
                    root=root,
                    block=str(block),
                    composite=composite,
                    atomic_a=atomic_a,
                    atomic_b=atomic_b,
                    top_k=int(args.top_k),
                )
            )

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "composite",
        "atomic_a",
        "atomic_b",
        "block",
        "top_k",
        "n_composite",
        "n_atomic_a",
        "n_atomic_b",
        "overlap_with_a",
        "overlap_with_b",
        "union_coverage",
        "new_feature_ratio",
        "atomic_overlap",
        "missing",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    print(f"[composition-overlap] wrote {out_path}")
    for row in rows:
        print(
            f"{row['composite']} block={block_short_name(str(row['block']))} "
            f"union={float(row['union_coverage']):.3f} new={float(row['new_feature_ratio']):.3f} "
            f"missing={row['missing']}"
        )


if __name__ == "__main__":
    main()
