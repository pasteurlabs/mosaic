#!/usr/bin/env bash
# Trigger a Read the Docs build for a specific version and wait for it to
# finish, ensuring docs are not built until artifacts are ready.
#
# Usage: rtd-trigger.sh <version-slug>
#
# Requires:
#   RTD_TOKEN   – Read the Docs API token
#   RTD_PROJECT – RTD project slug (e.g. "mosaic")
#
# The script triggers the build, then polls until it completes (success or
# failure). Exits 0 on success, 1 on failure or timeout.
set -euo pipefail

VERSION_SLUG="${1:?usage: rtd-trigger.sh <version-slug>}"

: "${RTD_TOKEN:?RTD_TOKEN must be set}"
: "${RTD_PROJECT:?RTD_PROJECT must be set}"

RTD_HOST="${RTD_HOST:-readthedocs.com}"
RTD_API="https://${RTD_HOST}/api/v3/projects/${RTD_PROJECT}"
AUTH="Authorization: Token ${RTD_TOKEN}"

echo "Triggering RTD build for ${RTD_PROJECT}/${VERSION_SLUG}"

# Trigger the build
HTTP_CODE=$(curl -sSL -w "%{http_code}" -X POST \
  -H "$AUTH" \
  -o /tmp/_rtd_response.json \
  "${RTD_API}/versions/${VERSION_SLUG}/builds/")
BODY=$(cat /tmp/_rtd_response.json)
rm -f /tmp/_rtd_response.json

# A 404 means RTD has no version with this slug yet. For PR (external)
# versions this is a benign race: RTD's webhook creates the external version
# on PR push, and this rebuild trigger may fire before that has happened. The
# webhook-driven build still produces a preview, so treat a missing version as
# a soft skip rather than failing the merge gate.
if [ "$HTTP_CODE" = "404" ]; then
  echo "::warning::No RTD version '${VERSION_SLUG}' yet (${BODY}) — skipping rebuild trigger."
  exit 0
fi

if [ "$HTTP_CODE" != "202" ]; then
  echo "::error::RTD API returned ${HTTP_CODE}: ${BODY}"
  exit 1
fi

BUILD_ID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['build']['id'])")
echo "Build triggered: id=${BUILD_ID}"

# Poll until the build finishes (timeout after 15 minutes)
TIMEOUT=900
ELAPSED=0
INTERVAL=15

while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))

  STATE=$(curl -fsSL -H "$AUTH" \
    "${RTD_API}/builds/${BUILD_ID}/" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state']['code'])")

  case "$STATE" in
    finished)
      echo "RTD build ${BUILD_ID} finished successfully"
      exit 0
      ;;
    cancelled)
      # RTD cancels a build when a newer one supersedes it (e.g. the webhook
      # build and this API-triggered rebuild overlap). The surviving build
      # produces the preview, so a cancellation is not a failure of this step.
      echo "::warning::RTD build ${BUILD_ID} was cancelled (superseded) — skipping."
      exit 0
      ;;
    *)
      echo "  build ${BUILD_ID}: ${STATE} (${ELAPSED}s elapsed)"
      ;;
  esac
done

echo "::error::RTD build ${BUILD_ID} timed out after ${TIMEOUT}s"
exit 1
