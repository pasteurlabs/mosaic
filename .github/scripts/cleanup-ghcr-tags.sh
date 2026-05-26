#!/usr/bin/env bash
# Delete stale :<sha> image tags from GHCR.
#
# Keeps :latest and any tag younger than MAX_AGE_DAYS.
# Requires GH_TOKEN with packages:write scope.
#
# Usage:
#   GH_TOKEN=... bash .github/scripts/cleanup-ghcr-tags.sh [max_age_days]

set -euo pipefail

MAX_AGE_DAYS="${1:-7}"
ORG="${GITHUB_REPOSITORY_OWNER:?must be set}"
CUTOFF=$(date -u -d "${MAX_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
      || date -u -v-"${MAX_AGE_DAYS}"d +%Y-%m-%dT%H:%M:%SZ)  # GNU || BSD date

echo "Cleaning GHCR images for ${ORG}/mosaic/* older than ${MAX_AGE_DAYS} days (before ${CUTOFF})"

# List all container packages under the mosaic namespace
PACKAGES=$(gh api --paginate "/orgs/${ORG}/packages?package_type=container" \
  --jq '.[].name | select(startswith("mosaic/"))')

DELETED=0
for PKG in $PACKAGES; do
  echo "Package: ${PKG}"

  # List versions, filter to sha-like tags older than cutoff
  gh api --paginate "/orgs/${ORG}/packages/container/${PKG}/versions" \
    --jq '.[] | select(.metadata.container.tags | length > 0) | {id, tags: .metadata.container.tags, updated: .updated_at}' \
  | jq -c "select(.updated < \"${CUTOFF}\") | select(.tags | all(test(\"^[0-9a-f]{7,40}$\")))" \
  | while read -r VERSION; do
      VID=$(echo "$VERSION" | jq -r '.id')
      TAGS=$(echo "$VERSION" | jq -r '.tags | join(", ")')
      echo "  Deleting version ${VID} (tags: ${TAGS})"
      gh api --method DELETE "/orgs/${ORG}/packages/container/${PKG}/versions/${VID}" || true
      DELETED=$((DELETED + 1))
    done
done

echo "Done. Deleted ${DELETED} version(s)."
