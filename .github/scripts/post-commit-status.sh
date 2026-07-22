#!/usr/bin/env bash
# Post a commit status to a PR head SHA.
#
# Usage: post-commit-status.sh <sha> <state> [description] [target-url]
#   state ∈ {pending, success, failure, error}
#
# The status context is fixed to "bench-ok" — the name branch protection
# requires. Splitting benchmark execution across two workflows (a fork-safe
# `pull_request` planner and a trusted `workflow_run` executor) means the real
# pass/fail verdict is posted here from the trusted run, while the check keeps
# the single stable name the required-checks list already knows.
#
# Requires GH_TOKEN with `statuses: write`. In the trusted workflow_run context
# the default GITHUB_TOKEN has it; in the fork `pull_request` run it can still
# write a status to its own head SHA (pending / short-circuit success).
#
# The SHA is fork-controlled (it arrives via a handoff artifact), so validate it
# is a full 40-hex commit id before using it in the API path — same defensive
# posture as the numeric PR-number guard in the workflow_run handoff.

set -euo pipefail

SHA="$1"
STATE="$2"
DESCRIPTION="${3:-}"
TARGET_URL="${4:-}"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: refusing to post status — '$SHA' is not a 40-hex commit SHA." >&2
  exit 1
fi

case "$STATE" in
  pending | success | failure | error) ;;
  *)
    echo "ERROR: invalid state '$STATE' (want pending|success|failure|error)." >&2
    exit 1
    ;;
esac

# GitHub caps status descriptions at 140 chars; trim defensively.
DESCRIPTION="${DESCRIPTION:0:140}"

ARGS=(
  --method POST
  "repos/${GITHUB_REPOSITORY}/statuses/${SHA}"
  -f "state=${STATE}"
  -f "context=bench-ok"
)
[[ -n "$DESCRIPTION" ]] && ARGS+=(-f "description=${DESCRIPTION}")
[[ -n "$TARGET_URL" ]] && ARGS+=(-f "target_url=${TARGET_URL}")

echo "Posting bench-ok=${STATE} to ${SHA}"
gh api "${ARGS[@]}"
