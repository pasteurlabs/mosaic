#!/usr/bin/env bash
# Delete stale :<sha> image tags from GHCR.
#
# A package version is deleted only if
#   - every tag on it is a commit SHA (so :latest is never touched),
#   - it was last updated more than MAX_AGE_DAYS ago,
#   - none of its tags is a main commit from the last MAX_AGE_DAYS days, and
#   - none of its tags is the base commit of an open PR (PR benchmarks pull
#     unchanged solvers at their base SHA).
# The version date alone is not enough: a commit that leaves a solver
# unchanged adds its tag to the existing version without bumping updated_at.
#
# Only packages of solvers under mosaic/tesseracts/ are cleaned.
#
# Requires GH_TOKEN with packages:write, contents:read and pull-requests:read.
# Set DRY_RUN=1 to list what would be deleted without deleting it.
#
# Usage:
#   GH_TOKEN=... GITHUB_REPOSITORY=owner/repo [DRY_RUN=1] \
#     bash .github/scripts/cleanup-ghcr-tags.sh [max_age_days]

set -euo pipefail

MAX_AGE_DAYS="${1:-7}"
REPO="${GITHUB_REPOSITORY:?must be set}"
ORG="${REPO%%/*}"
DRY_RUN="${DRY_RUN:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUTOFF=$(date -u -d "${MAX_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
      || date -u -v-"${MAX_AGE_DAYS}"d +%Y-%m-%dT%H:%M:%SZ)  # GNU || BSD date

echo "Cleaning GHCR images for ${ORG}/mosaic/* older than ${MAX_AGE_DAYS} days (before ${CUTOFF})"
if [[ -n "$DRY_RUN" ]]; then
  echo "Dry run: nothing will be deleted"
fi

KEEP_SHAS=$(
  {
    gh api --paginate "/repos/${REPO}/commits?since=${CUTOFF}&per_page=100" --jq '.[].sha'
    gh api --paginate "/repos/${REPO}/pulls?state=open&per_page=100" --jq '.[].base.sha'
  } | jq -Rn '[inputs] | unique'
)
echo "Keeping $(jq length <<<"$KEEP_SHAS") recent main / open-PR base SHA(s)"

PACKAGES=$(find "${ROOT}/mosaic/tesseracts" -name tesseract_config.yaml \
  -exec sed -n 's/^name:[[:space:]]*//p' {} + | sort -u)

DELETED=0
FAILED=0
for PKG in $PACKAGES; do
  echo "Package: mosaic/${PKG}"
  VERSIONS_PATH="/orgs/${ORG}/packages/container/mosaic%2F${PKG}/versions"

  if ! VERSIONS=$(gh api --paginate "$VERSIONS_PATH" \
      --jq '.[] | {id, tags: .metadata.container.tags, updated: .updated_at}'); then
    echo "  Failed to list versions"
    FAILED=$((FAILED + 1))
    continue
  fi

  while read -r VERSION; do
    VID=$(jq -r '.id' <<<"$VERSION")
    TAGS=$(jq -r '.tags | join(", ")' <<<"$VERSION")
    if [[ -n "$DRY_RUN" ]]; then
      echo "  Would delete version ${VID} (tags: ${TAGS})"
    elif gh api --silent --method DELETE "${VERSIONS_PATH}/${VID}"; then
      echo "  Deleted version ${VID} (tags: ${TAGS})"
    else
      echo "  Failed to delete version ${VID} (tags: ${TAGS})"
      FAILED=$((FAILED + 1))
      continue
    fi
    DELETED=$((DELETED + 1))
  done < <(jq -c --arg cutoff "$CUTOFF" --argjson keep "$KEEP_SHAS" '
    select(.updated < $cutoff)
    | select(.tags | length > 0 and all(test("^[0-9a-f]{7,40}$")))
    | select(.tags - $keep == .tags)
  ' <<<"$VERSIONS")
done

VERB="Deleted"
if [[ -n "$DRY_RUN" ]]; then
  VERB="Would delete"
fi
echo "Done. ${VERB} ${DELETED} version(s), ${FAILED} failure(s)."
[[ "$FAILED" -eq 0 ]]
