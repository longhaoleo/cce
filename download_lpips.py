"""Download LPIPS/AlexNet weights used by evaluation scripts.

Default Torch cache:

    /root/autodl-tmp/models/torch

The LPIPS package uses Torch/TorchVision cache paths internally. This script
sets TORCH_HOME to the same directory used by evaluation, then instantiates the
Alex LPIPS model once to prefetch the required weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_TORCH_HOME = "/root/autodl-tmp/models/torch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LPIPS/AlexNet weights for local evaluation.")
    parser.add_argument("--torch_home", type=str, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_home = Path(args.torch_home).expanduser().resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    import lpips

    print(f"[lpips] TORCH_HOME={torch_home}")
    print(f"[lpips] loading LPIPS(net={args.net!r})")
    _ = lpips.LPIPS(net=str(args.net))

    manifest = {
        "net": str(args.net),
        "torch_home": str(torch_home),
    }
    manifest_path = torch_home / "lpips_download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[lpips] wrote {manifest_path}")
    print("[lpips] done")


if __name__ == "__main__":
    main()
