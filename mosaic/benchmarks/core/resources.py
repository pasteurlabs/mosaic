"""Container resource-limit helpers."""

from __future__ import annotations

import os

# Fraction of total host RAM to expose to each Tesseract container.
# Override via MOSAIC_CONTAINER_MEM_FRACTION (0–1, default 0.9).
_DEFAULT_MEM_FRACTION = 0.9


def container_memory_args() -> list[str]:
    """Return Docker CLI args that cap container memory at a fraction of host RAM.

    Reads ``MOSAIC_CONTAINER_MEM_FRACTION`` (default 0.9) and uses psutil to
    query total physical memory.  Returns e.g. ``["--memory", "28g"]``.

    Returns an empty list if psutil is unavailable or the env var is set to 0.
    """
    frac_str = os.environ.get("MOSAIC_CONTAINER_MEM_FRACTION", "")
    frac = float(frac_str) if frac_str else _DEFAULT_MEM_FRACTION
    if frac <= 0:
        return []

    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
    except Exception:
        return []

    limit_bytes = int(total_bytes * min(frac, 1.0))
    # Docker --memory accepts bytes directly.
    return ["--memory", str(limit_bytes)]
