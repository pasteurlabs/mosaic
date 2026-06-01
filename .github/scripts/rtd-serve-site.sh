#!/usr/bin/env bash
# Populate Read the Docs' HTML output from a pre-rendered docs site artifact.
#
# The docs are rendered on GitHub Actions (benchmark.yml / publish-results.yml),
# where the full mosaic/jax stack is installed and memory is ample, then
# uploaded as a `docs-site-<sha>` artifact. RTD's constrained builders OOM-kill
# the heavy `import mosaic`, so RTD no longer renders anything — it just
# downloads the rendered _site/ and serves it.
#
# Usage: rtd-serve-site.sh <output-dir>
#   <output-dir> is $READTHEDOCS_OUTPUT/html.
#
# Requires (as RTD environment variables):
#   GITHUB_TOKEN  – token with actions:read on the repo
#   GITHUB_REPO   – owner/repo (defaults to pasteurlabs/mosaic)
#
# RTD-provided env used to pick the right artifact:
#   READTHEDOCS_VERSION_TYPE        – external | branch | tag
#   READTHEDOCS_GIT_COMMIT_HASH     – commit being built
#   READTHEDOCS_VERSION_NAME        – version/tag name
set -euo pipefail

OUTPUT_DIR="${1:?usage: rtd-serve-site.sh <output-dir>}"
export GITHUB_REPO="${GITHUB_REPO:-pasteurlabs/mosaic}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING="$(mktemp -d)"
mkdir -p "$OUTPUT_DIR"

placeholder() {
  # Minimal static page shown when no rendered site artifact exists yet
  # (e.g. an early PR build that fired before the benchmark run finished).
  local msg="$1"
  cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>Mosaic docs — building</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto; padding: 0 1rem; color: #222; }
    .spinner { color: #2780e3; }
  </style>
</head>
<body>
  <h1><span class="spinner">⏳</span> Documentation is building</h1>
  <p>${msg}</p>
  <p>This page refreshes automatically. The preview will update once the
     GitHub Actions run finishes rendering and uploads the site.</p>
</body>
</html>
HTML
  echo "Wrote placeholder page: ${msg}"
}

if [ -z "$GITHUB_TOKEN" ]; then
  placeholder "GITHUB_TOKEN is not configured on Read the Docs, so the rendered site cannot be fetched."
  exit 0
fi

# Pick the artifact name matching this build. PR (external) builds key off
# the commit SHA (benchmark.yml uploads docs-site-<sha>); branch/tag builds
# use stable names (publish-results.yml uploads docs-site-latest and
# docs-site-release-<version>).
#   - tag (release):   docs-site-release-<version>, then docs-site-latest
#   - external (PR):   docs-site-<sha>
#   - branch (latest): docs-site-latest, then docs-site-<sha>
CANDIDATES=()
case "${READTHEDOCS_VERSION_TYPE:-}" in
  tag)
    [ -n "${READTHEDOCS_VERSION_NAME:-}" ] && CANDIDATES+=("docs-site-release-${READTHEDOCS_VERSION_NAME}")
    CANDIDATES+=("docs-site-latest")
    ;;
  external)
    [ -n "${READTHEDOCS_GIT_COMMIT_HASH:-}" ] && CANDIDATES+=("docs-site-${READTHEDOCS_GIT_COMMIT_HASH}")
    ;;
  *)
    CANDIDATES+=("docs-site-latest")
    [ -n "${READTHEDOCS_GIT_COMMIT_HASH:-}" ] && CANDIDATES+=("docs-site-${READTHEDOCS_GIT_COMMIT_HASH}")
    ;;
esac

FETCHED=false
for name in "${CANDIDATES[@]}"; do
  echo "Trying docs site artifact '${name}'"
  if bash "$SCRIPT_DIR/fetch-artifact.sh" "$name" "$STAGING"; then
    FETCHED=true
    break
  fi
done

if [ "$FETCHED" != "true" ]; then
  placeholder "The rendered docs for this build are not available yet (no <code>${CANDIDATES[0]:-docs-site}</code> artifact found)."
  exit 0
fi

# The artifact contains the rendered _site/ contents at its top level.
echo "Copying rendered site into ${OUTPUT_DIR}"
cp -r "$STAGING"/. "$OUTPUT_DIR"/
rm -rf "$STAGING"

if [ ! -f "$OUTPUT_DIR/index.html" ]; then
  echo "::warning::Fetched artifact has no index.html at its root"
  placeholder "The fetched docs artifact did not contain an index.html."
fi
echo "Docs site is ready in ${OUTPUT_DIR}"
