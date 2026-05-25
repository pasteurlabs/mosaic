#!/usr/bin/env bash
# Build the `dealii-root:latest` wrapper image required by this tesseract.
#
# The upstream `dealii/dealii:latest` image defaults to USER `dealii`, which
# breaks the tesseract template's hardcoded `apt-get` steps (EACCES on
# /var/lib/apt/lists/partial in both build_stage and run_stage). This wrapper
# does nothing but switch to USER root so the tesseract build can proceed.
#
# Shared with mosaic/tesseracts/thermal-mesh/dealii-heat (same base image).
# Idempotent: if the wrapper image already exists, this is essentially a no-op.

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

cat > "$WORKDIR/Dockerfile" <<'EOF'
FROM dealii/dealii:v9.7.1-jammy
USER root
EOF

docker build -t dealii-root:latest "$WORKDIR"
