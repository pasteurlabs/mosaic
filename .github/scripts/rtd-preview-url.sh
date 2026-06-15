#!/usr/bin/env bash
# Resolve the public Read the Docs documentation URL for a PR's preview build
# and write it to $GITHUB_OUTPUT as `preview_url`.
#
# Usage: rtd-preview-url.sh <version-slug>
#   <version-slug> is the PR number (RTD's external version slug).
#
# Requires:
#   RTD_TOKEN   – Read the Docs API token
#   RTD_PROJECT – RTD project slug (e.g. "pasteur-labs-mosaic")
#
# The external version already exists by the time this runs (RTD's webhook
# creates it on PR push), so the lookup does not depend on any in-flight build.
# Best-effort: any failure or missing URL leaves `preview_url` empty so callers
# can simply omit the preview link.
set -euo pipefail

VERSION_SLUG="${1:?usage: rtd-preview-url.sh <version-slug>}"

: "${RTD_TOKEN:?RTD_TOKEN must be set}"
: "${RTD_PROJECT:?RTD_PROJECT must be set}"

RTD_HOST="${RTD_HOST:-readthedocs.com}"
RTD_API="https://${RTD_HOST}/api/v3/projects/${RTD_PROJECT}"
AUTH="Authorization: Token ${RTD_TOKEN}"

DOC_URL=$(curl -fsSL -H "$AUTH" \
  "${RTD_API}/versions/${VERSION_SLUG}/" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('urls',{}).get('documentation',''))" \
  2>/dev/null || true)

if [ -n "$DOC_URL" ]; then
  echo "Docs preview URL: ${DOC_URL}"
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "preview_url=${DOC_URL}" >> "$GITHUB_OUTPUT"
else
  echo "::warning::Could not resolve RTD preview URL for version '${VERSION_SLUG}' — omitting preview link."
fi
