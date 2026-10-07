#!/usr/bin/env bash
set -euo pipefail
campaign=$1
export UV_CACHE_DIR=/data/personal/andrinr/uv-cache
uv run --no-project --with numpy --with matplotlib python "$campaign/report.py" "$campaign"
