"""SharedSAE 主线的稳定默认值。"""

from __future__ import annotations


DEFAULT_BLOCKS = (
    "unet.down_blocks.2.attentions.1",
    "unet.mid_block.attentions.0",
    "unet.up_blocks.0.attentions.0",
    "unet.up_blocks.0.attentions.1",
)

