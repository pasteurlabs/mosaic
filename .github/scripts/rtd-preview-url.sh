#!/usr/bin/env bash
# Resolve the Read the Docs PR-preview documentation URL and write it to
# $GITHUB_OUTPUT as `preview_url`.
#
# Usage: rtd-preview-url.sh <pr-number>
#   <pr-number> is RTD's external version slug for the PR.
#
# Requires:
#   RTD_TOKEN   – Read the Docs API token
#   RTD_PROJECT – RTD project slug (e.g. "pasteur-labs-mosaic")
#
# Why this constructs the URL instead of looking up the version:
#   The external (PR) version is NOT enumerable through the versions API for
#   this project — /versions/<slug>/ 404s and /versions/?type=external returns
#   count 0 (external versions aren't exposed to the CI token, and on a
#   subproject they belong to the parent's external domain anyway). But the
#   preview URL is deterministic, so we build it from data the API *does*
#   return:
#
#     https://<parent-slug>--<PR>.com.readthedocs.build/projects/<alias>/<PR>/
#
#   where <parent-slug> is RTD_PROJECT's superproject and <alias> is the
#   subproject alias the parent mounts it under. For a standalone project
#   (no superproject) the form is:
#
#     https://<project-slug>--<PR>.com.readthedocs.build/<PR>/
#
#   The preview is auth-gated (private repo → GitHub OAuth), so an
#   unauthenticated GET 302-redirects to the RTD login; a logged-in reviewer
#   sees the rendered docs. We therefore do NOT verify the URL with curl (a 302
#   is success, not failure) — we just emit the constructed link.
#
# Best-effort: any failure leaves `preview_url` empty so callers can omit the
# preview link.
set -euo pipefail

PR_NUMBER="${1:?usage: rtd-preview-url.sh <pr-number>}"

: "${RTD_TOKEN:?RTD_TOKEN must be set}"
: "${RTD_PROJECT:?RTD_PROJECT must be set}"

RTD_HOST="${RTD_HOST:-readthedocs.com}"
# RTD's external (PR preview) builds are served from a dedicated domain,
# distinct from the production docs domain.
RTD_EXTERNAL_DOMAIN="${RTD_EXTERNAL_DOMAIN:-com.readthedocs.build}"
RTD_API="https://${RTD_HOST}/api/v3/projects"
AUTH="Authorization: Token ${RTD_TOKEN}"

# curl that captures the HTTP status, echoing the body on 200 and logging a
# warning otherwise (this step is best-effort / continue-on-error).
_api_get() {
  local url="$1" body code
  body=$(mktemp)
  code=$(curl -sSL -w "%{http_code}" -o "$body" -H "$AUTH" "$url" || echo "000")
  if [ "$code" = "200" ]; then
    cat "$body"
  else
    echo "::warning::RTD GET ${url} returned HTTP ${code}: $(head -c 300 "$body")" >&2
  fi
  rm -f "$body"
}

# Parent (super)project slug, or empty for a standalone project.
PARENT_SLUG=$(_api_get "${RTD_API}/${RTD_PROJECT}/" \
  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("subproject_of") or {}).get("slug") or "")' \
  2>/dev/null || true)

if [ -n "$PARENT_SLUG" ]; then
  # Subproject: the external host uses the PARENT slug, and the path includes
  # the subproject alias the parent mounts this project under.
  ALIAS=$(RTD_PROJECT="$RTD_PROJECT" _api_get "${RTD_API}/${PARENT_SLUG}/subprojects/" \
    | python3 -c '
import json, os, sys
want = os.environ["RTD_PROJECT"]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for s in data.get("results") or []:
    child = s.get("child") or {}
    child_slug = child.get("slug") if isinstance(child, dict) else child
    if child_slug == want:
        print(s.get("alias") or "")
        break
' 2>/dev/null || true)
  if [ -n "$ALIAS" ]; then
    HOST="${PARENT_SLUG}--${PR_NUMBER}.${RTD_EXTERNAL_DOMAIN}"
    PREVIEW_URL="https://${HOST}/projects/${ALIAS}/${PR_NUMBER}/"
  fi
else
  # Standalone project: host uses the project slug, no /projects/<alias> prefix.
  HOST="${RTD_PROJECT}--${PR_NUMBER}.${RTD_EXTERNAL_DOMAIN}"
  PREVIEW_URL="https://${HOST}/${PR_NUMBER}/"
fi

if [ -n "${PREVIEW_URL:-}" ]; then
  echo "Docs preview URL: ${PREVIEW_URL}"
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "preview_url=${PREVIEW_URL}" >> "$GITHUB_OUTPUT"
else
  echo "::warning::Could not construct RTD preview URL for PR '${PR_NUMBER}' (project '${RTD_PROJECT}') — omitting preview link."
fi
