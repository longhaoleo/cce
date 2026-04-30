#!/usr/bin/env python3
"""一键下载 SDXL Base 到 ~/autodl-tmp/models。"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LOCAL_DIR = Path("~/autodl-tmp/models/stabilityai/stable-diffusion-xl-base-1.0").expanduser()

LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=MODEL_ID, local_dir=str(LOCAL_DIR))

print(f"下载完成: {LOCAL_DIR}")
