#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a field from a tesseract_config.yaml file.

Supports dotted paths for nested keys (e.g. ``build_config.base_image``).
Prints the value to stdout, or an empty string if the key is missing.

Usage:
    python .github/scripts/tesseract-config-field.py path/to/tesseract_config.yaml name
    python .github/scripts/tesseract-config-field.py path/to/tesseract_config.yaml build_config.base_image
"""

from __future__ import annotations

import sys

import yaml


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <config.yaml> <dotted.key>", file=sys.stderr)
        sys.exit(1)

    config_path, dotted_key = sys.argv[1], sys.argv[2]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    value = cfg
    for part in dotted_key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    print(value if value is not None else "")


if __name__ == "__main__":
    main()
