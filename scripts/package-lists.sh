#!/usr/bin/env bash

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Write the package lists of the solver images, one text file per image.
#
# Usage: scripts/package-lists.sh OUT_DIR [IMAGE...]
#
# Without IMAGEs, lists <name>:latest for every tesseract_config.yaml under
# mosaic/tesseracts. Each file holds the image ID, the Python packages in
# /python-env, the conda environment where /python-env is one, the Julia
# project where /app/julia_env exists, and the apt packages. Header lines
# start with "# "; to check that two sets of images match, compare the rest:
#
#   diff <(grep -hv '^# ' a/*.txt) <(grep -hv '^# ' b/*.txt)
#
# The containers run without network access.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 OUT_DIR [IMAGE...]" >&2
    exit 2
fi
out=$1
shift
mkdir -p "$out"

if [ $# -eq 0 ]; then
    root=$(cd "$(dirname "$0")/.." && pwd)
    set -- $(sed -n 's/^name: *//p' "$root"/mosaic/tesseracts/*/*/tesseract_config.yaml | sed 's/$/:latest/')
fi

for image in "$@"; do
    file="$out/${image%%:*}.txt"
    {
        echo "# image: $image"
        echo "# id: $(docker image inspect -f '{{.Id}}' "$image")"
        echo "# created: $(docker image inspect -f '{{.Created}}' "$image")"
        docker run --rm --network none --entrypoint sh "$image" -c '
            echo; echo "## pip freeze --all (/python-env)"
            /python-env/bin/python -m pip freeze --all 2>/dev/null | LC_ALL=C sort
            if [ -d /python-env/conda-meta ]; then
                echo; echo "## conda list --explicit (/python-env)"
                conda list --explicit -p /python-env
            fi
            for f in Project.toml Manifest.toml LocalPreferences.toml; do
                if [ -f "/app/julia_env/$f" ]; then
                    echo; echo "## /app/julia_env/$f"; cat "/app/julia_env/$f"
                fi
            done
            echo; echo "## dpkg-query -W"
            dpkg-query -W -f "\${Package} \${Version}\n" | LC_ALL=C sort
        '
    } >"$file"
    echo "$file"
done
