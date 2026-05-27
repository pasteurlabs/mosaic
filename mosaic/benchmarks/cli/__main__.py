# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m mosaic.benchmarks.cli`` to invoke the Typer app."""

from __future__ import annotations

from mosaic.benchmarks.cli import app

if __name__ == "__main__":
    app()
