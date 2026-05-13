#!/usr/bin/env bash
# Build the `firedrake-root:latest` wrapper image required by this tesseract.
#
# The upstream `firedrakeproject/firedrake` image defaults to a non-root user,
# which breaks the tesseract template's `apt-get` steps (EACCES on
# /var/lib/apt/lists/partial). This wrapper switches to USER root.

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

cat > "$WORKDIR/Dockerfile" <<'EOF'
FROM firedrakeproject/firedrake:2024-11
USER root
EOF

docker build -t firedrake-root:latest "$WORKDIR"
