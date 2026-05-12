#!/usr/bin/env python3
"""Validate all tesseract_config.yaml files under mosaic/tesseracts/.

Checks required fields, value constraints, and basic structure.
Exits non-zero if any errors are found.

Usage (in CI):
    python .github/scripts/validate-tesseract-configs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

errors: list[str] = []
configs = sorted(Path("mosaic/tesseracts").rglob("tesseract_config.yaml"))

if not configs:
    print("ERROR: no tesseract_config.yaml files found")
    sys.exit(1)

for p in configs:
    try:
        with open(p) as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            errors.append(f"{p}: not a YAML mapping")
            continue
        if "name" not in doc:
            errors.append(f"{p}: missing 'name' field")
        metadata = doc.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{p}: missing 'metadata' block")
            continue
        mosaic_meta = metadata.get("mosaic")
        if not isinstance(mosaic_meta, dict):
            errors.append(f"{p}: missing 'metadata.mosaic' block")
            continue
        if not mosaic_meta.get("name"):
            errors.append(f"{p}: mosaic.name is required")
        if not mosaic_meta.get("backend"):
            errors.append(f"{p}: mosaic.backend is required")
        ad = mosaic_meta.get("ad_strategy")
        if ad is not None and ad not in ("autodiff", "adjoint", "hybrid", "null"):
            errors.append(
                f"{p}: mosaic.ad_strategy={ad!r} not in (autodiff, adjoint, hybrid, null)"
            )
        color = mosaic_meta.get("color")
        if color is not None:
            if not re.match(r"^#[0-9a-fA-F]{6}$", color):
                errors.append(f"{p}: mosaic.color={color!r} is not valid hex (#RRGGBB)")
        for bool_field in ("uses_gpu", "differentiable"):
            val = mosaic_meta.get(bool_field)
            if val is not None and not isinstance(val, bool):
                errors.append(f"{p}: mosaic.{bool_field}={val!r} must be a boolean")
    except Exception as e:
        errors.append(f"{p}: {e}")

print(f"Checked {len(configs)} tesseract configs")
if errors:
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
print("All valid.")
