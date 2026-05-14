"""Download CLIP weights used by evaluation scripts.

Default location:

    /root/autodl-tmp/models/clip-vit-base-patch32

The evaluation scripts prefer this local directory when it exists. This keeps
CLIP evaluation reproducible and avoids repeated HuggingFace downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/models/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CLIP weights for local evaluation.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor

    print(f"[clip] loading {args.model_id}")
    model = CLIPModel.from_pretrained(str(args.model_id), local_files_only=bool(args.local_files_only))
    processor = CLIPProcessor.from_pretrained(str(args.model_id), local_files_only=bool(args.local_files_only))

    print(f"[clip] saving to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    manifest = {
        "model_id": str(args.model_id),
        "output_dir": str(output_dir),
    }
    with (output_dir / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[clip] done")


if __name__ == "__main__":
    main()
