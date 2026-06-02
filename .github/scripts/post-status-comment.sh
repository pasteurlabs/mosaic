#!/usr/bin/env bash
# Post or update a Mosaic status comment on a PR.
#
# Usage: post-status-comment.sh <pr-number> <markdown-file>
#
# Finds an existing comment by the bot with "## Mosaic" header and updates it,
# or creates a new one. Requires GH_TOKEN with write permissions.

set -euo pipefail

PR_NUMBER="$1"
MD_FILE="$2"

if [[ ! -f "$MD_FILE" ]]; then
  echo "ERROR: markdown file not found: $MD_FILE" >&2
  exit 1
fi

MARKER="<!-- mosaic-benchmark-bot -->"
BODY="${MARKER}
$(cat "$MD_FILE")"

# Find existing comment by marker (suppress stderr so a 401 doesn't
# leak garbage into COMMENT_ID; strip \r that Windows-style line endings
# or gh pagination may introduce).
COMMENT_ID=$(gh api \
  "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
  --paginate --jq ".[] | select(.body | startswith(\"${MARKER}\")) | .id" \
  2>/dev/null | tr -d '\r' | head -n1 || true)

if [[ "$COMMENT_ID" =~ ^[0-9]+$ ]]; then
  echo "Updating existing comment ${COMMENT_ID}"
  gh api \
    --method PATCH \
    "repos/${GITHUB_REPOSITORY}/issues/comments/${COMMENT_ID}" \
    -f body="$BODY"
else
  echo "Creating new comment on PR #${PR_NUMBER}"
  gh api \
    --method POST \
    "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
    -f body="$BODY"
fi
