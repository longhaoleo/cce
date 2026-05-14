"""Pixel and intervention-diagnostic metrics for SharedSAE batch outputs."""

from __future__ import annotations

import sys

from .run_batch_metrics import main


if __name__ == "__main__":
    if "--metrics" not in sys.argv:
        sys.argv.extend(["--metrics", "pixel,diag"])
    main()
