#!/usr/bin/env bash
# Post or update a Mosaic status comment on a PR.
#
# Usage: post-status-comment.sh <pr-number> [markdown-file]
#
# Finds an existing comment by the bot with "## Mosaic" header and updates it,
# or creates a new one. Requires GH_TOKEN with write permissions.
#
# The markdown file (the benchmark status report) is optional: PRs that ran no
# benchmarks (benchmark:none, docs-only) skip it, since the report would just be
# an all-zero diff against the baseline. In that case the comment is only the
# docs-preview banner.
#
# When DOCS_PREVIEW_URL is set, a prominent banner linking to the rendered
# Read the Docs preview is prepended above the status report. That preview is
# the primary way to view benchmark *results* (the embedded plots), whereas the
# report below only summarises pass/fail status — so the link goes first.

set -euo pipefail

PR_NUMBER="$1"
MD_FILE="${2:-}"

REPORT=""
if [[ -n "$MD_FILE" ]]; then
  if [[ ! -f "$MD_FILE" ]]; then
    echo "ERROR: markdown file not found: $MD_FILE" >&2
    exit 1
  fi
  REPORT="$(cat "$MD_FILE")"
fi

MARKER="<!-- mosaic-benchmark-bot -->"

PREVIEW_BANNER=""
if [[ -n "${DOCS_PREVIEW_URL:-}" ]]; then
  # Deep-link straight to the results landing page, which fans out to every
  # domain's plots. The Quarto project root is the repo root and results.qmd
  # lives under docs/, so it renders to docs/results.html (only index.qmd is
  # promoted to the site root; every other page keeps its docs/ prefix).
  RESULTS_URL="${DOCS_PREVIEW_URL%/}/docs/results.html"
  # Tailor the blurb to whether a status report follows.
  if [[ -n "$REPORT" ]]; then
    PREVIEW_BLURB="The rendered docs preview has every plot for this run (forward accuracy,
gradients, cost, optimization) merged with existing baseline results on \`main\`. The summary below reports pass/fail status."
  else
    PREVIEW_BLURB="No benchmarks ran for this PR, so there is no status report."
  fi
  PREVIEW_BANNER="### 📊 [**View the full benchmark results**](${RESULTS_URL})

${PREVIEW_BLURB}

---
"
fi

# Nothing to say (no report and no preview link) → don't post an empty comment.
if [[ -z "$REPORT" && -z "$PREVIEW_BANNER" ]]; then
  echo "No status report and no docs-preview URL — skipping PR comment."
  exit 0
fi

BODY="${MARKER}
${PREVIEW_BANNER}${REPORT}"

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
