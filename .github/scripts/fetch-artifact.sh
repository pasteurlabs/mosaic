#!/usr/bin/env bash
# Download a GitHub Actions artifact by name into a target directory.
#
# Usage: fetch-artifact.sh <artifact-name> <target-dir>
#
# Requires:
#   GITHUB_TOKEN  – a token with actions:read (or repo) scope
#   GITHUB_REPO   – owner/repo (e.g. pasteurlabs/mosaic)
#
# Exits 0 on success, 1 on missing args/env, 2 if artifact not found (non-fatal).
set -euo pipefail

ARTIFACT_NAME="${1:?usage: fetch-artifact.sh <artifact-name> <target-dir>}"
TARGET_DIR="${2:?usage: fetch-artifact.sh <artifact-name> <target-dir>}"

: "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
: "${GITHUB_REPO:?GITHUB_REPO must be set}"

API="https://api.github.com/repos/${GITHUB_REPO}/actions/artifacts"
AUTH="Authorization: Bearer ${GITHUB_TOKEN}"

# Find the most recent artifact with the given name
LISTING=$(curl -sSL -w "\n%{http_code}" -H "$AUTH" \
  "${API}?name=${ARTIFACT_NAME}&per_page=1") || true
HTTP_CODE=$(echo "$LISTING" | tail -1)
LISTING_BODY=$(echo "$LISTING" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "GitHub API returned ${HTTP_CODE} when listing artifacts — continuing without results"
  exit 2
fi

ARTIFACT_ID=$(echo "$LISTING_BODY" | python3 -c "
import json, sys
data = json.load(sys.stdin)
arts = data.get('artifacts', [])
print(arts[0]['id'] if arts else '')
")

if [ -z "$ARTIFACT_ID" ]; then
  echo "No artifact '${ARTIFACT_NAME}' found — continuing without it"
  exit 2
fi

echo "Downloading artifact '${ARTIFACT_NAME}' (id=${ARTIFACT_ID})"
mkdir -p "$TARGET_DIR"
# -L follows the redirect to the temporary download URL
curl -sSL -H "$AUTH" -o /tmp/_artifact.zip \
  "${API}/${ARTIFACT_ID}/zip"
if [ ! -s /tmp/_artifact.zip ]; then
  echo "Download failed for artifact '${ARTIFACT_NAME}'"
  rm -f /tmp/_artifact.zip
  exit 2
fi
unzip -q -o /tmp/_artifact.zip -d "$TARGET_DIR"
rm -f /tmp/_artifact.zip
echo "Extracted '${ARTIFACT_NAME}' into ${TARGET_DIR}"
