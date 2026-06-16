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
#
# NB: the version *detail* endpoint (/versions/<slug>/) returns 404 for
# external (PR) versions on RTD — it only serves branch/tag versions — even
# though POSTing a build to /versions/<slug>/builds/ works (that's how
# rtd-trigger.sh succeeds). So resolve the URL via the versions *list* filtered
# to external versions and match the slug, falling back to the detail endpoint
# for non-external setups.
set -euo pipefail

VERSION_SLUG="${1:?usage: rtd-preview-url.sh <version-slug>}"

: "${RTD_TOKEN:?RTD_TOKEN must be set}"
: "${RTD_PROJECT:?RTD_PROJECT must be set}"

RTD_HOST="${RTD_HOST:-readthedocs.com}"
RTD_API="https://${RTD_HOST}/api/v3/projects/${RTD_PROJECT}"
AUTH="Authorization: Token ${RTD_TOKEN}"

# Extract urls.documentation for the version whose slug matches VERSION_SLUG.
# Accepts either a list response ({"results": [...]}) or a single version
# object, so it works for both the list and detail endpoints.
_extract_doc_url() {
  VERSION_SLUG="$VERSION_SLUG" python3 -c '
import json, os, sys
slug = os.environ["VERSION_SLUG"]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
versions = data.get("results") if isinstance(data, dict) and "results" in data else [data]
for v in versions or []:
    if not isinstance(v, dict):
        continue
    if str(v.get("slug")) == slug:
        print((v.get("urls") or {}).get("documentation", "") or "")
        break
' 2>/dev/null || true
}

# curl that captures the HTTP status, so a failure is logged rather than
# silently swallowed (this step is best-effort / continue-on-error).
_api_get() {
  local url="$1" body
  body=$(mktemp)
  local code
  code=$(curl -sSL -w "%{http_code}" -o "$body" -H "$AUTH" "$url" || echo "000")
  if [ "$code" = "200" ]; then
    cat "$body"
  else
    echo "::warning::RTD GET ${url} returned HTTP ${code}: $(head -c 300 "$body")" >&2
  fi
  rm -f "$body"
}

# Primary: list external versions and match the PR slug.
DOC_URL=$(_api_get "${RTD_API}/versions/?type=external&slug=${VERSION_SLUG}" | _extract_doc_url)

# Fallback: the version detail endpoint (works for branch/tag-style setups).
if [ -z "$DOC_URL" ]; then
  DOC_URL=$(_api_get "${RTD_API}/versions/${VERSION_SLUG}/" | _extract_doc_url)
fi

if [ -n "$DOC_URL" ]; then
  echo "Docs preview URL: ${DOC_URL}"
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "preview_url=${DOC_URL}" >> "$GITHUB_OUTPUT"
else
  echo "::warning::Could not resolve RTD preview URL for version '${VERSION_SLUG}' — omitting preview link."
fi
