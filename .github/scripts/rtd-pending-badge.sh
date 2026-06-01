#!/usr/bin/env bash
# Prepend a "benchmark results building" callout to each results page.
#
# Used by the Read the Docs build (see .readthedocs.yaml) when an external
# (PR) build runs before this PR's own benchmark-results-<sha> artifact is
# available. The committed results_*.qmd files still render with baseline
# content; this badge tells the reader the PR-specific results are on the way
# and that the preview will update automatically on the follow-up build
# triggered by benchmark.yml.
#
# Usage: rtd-pending-badge.sh
#   Operates on docs/results_*.qmd in the current working tree. Idempotent:
#   the badge is keyed by a marker comment and inserted at most once per file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARKER="<!-- rtd-pending-badge -->"

# The callout, inserted after the YAML front matter. Keep the marker on its
# own line as the idempotency key.
BADGE=$(cat <<EOF
${MARKER}
::: {.callout-warning title='Benchmark results are still building'}
The benchmark results for this pull request are still being generated.
The plots below show the current baseline; this preview updates
automatically once the PR run finishes.
:::
EOF
)

insert_badge() {
  # Stream the file, echoing the badge once, right after the second '---'
  # line that closes the YAML front matter.
  local f="$1"
  local fm=0 inserted=0 line
  while IFS= read -r line || [ -n "$line" ]; do
    printf '%s\n' "$line"
    if [ "$inserted" -eq 0 ] && [ "$line" = "---" ]; then
      fm=$((fm + 1))
      if [ "$fm" -eq 2 ]; then
        printf '\n%s\n' "$BADGE"
        inserted=1
      fi
    fi
  done < "$f"
}

shopt -s nullglob
for f in "$ROOT"/docs/results_*.qmd; do
  if grep -qF "$MARKER" "$f"; then
    continue
  fi
  insert_badge "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  echo "Added pending badge to $(basename "$f")"
done
