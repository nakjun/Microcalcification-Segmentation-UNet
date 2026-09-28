#!/usr/bin/env bash
# Link an already extracted original dataset, or copy required original runtime artifacts.
source "$(dirname "$0")/common.sh"
mode="${1:-}"; path="${2:-}"
case "$mode" in
 csaw|inbreast|cbis)
  [[ -d "$path" ]] || { echo 'Usage: load_data.sh [--execute] {csaw|inbreast|cbis} EXTRACTED_DIRECTORY' >&2; exit 2; }
  path="$(cd "$path" && pwd)"
  [[ ! -e "datasets/$mode" && ! -L "datasets/$mode" ]] || { echo 'Dataset destination already exists' >&2; exit 2; }
  run mkdir -p datasets
  run ln -s "$path" "datasets/$mode"
  echo 'Raw data path registered. Original manifests still require their recorded paths; this does not regenerate splits.'
  ;;
 artifacts)
  [[ -d "$path/runs" ]] || { echo 'Usage: load_data.sh [--execute] artifacts ORIGINAL_PROJECT_ROOT' >&2; exit 2; }
  path="$(cd "$path" && pwd)"
  [[ "$path" != "$STUDY_ROOT" ]] || { echo 'Source and destination must differ' >&2; exit 2; }
  # Copy, never symlink/hardlink mutable run trees. This can require substantial disk space.
  # Preserve historical hashes/records. Never overwrite an existing destination tree.
  for d in runs reports manifests docs; do
   [[ -d "$path/$d" ]] || continue
   [[ ! -e "$d" ]] || { echo "Destination exists: $d; refusing partial merge" >&2; exit 2; }
  done
  for d in runs reports manifests docs; do
   [[ -d "$path/$d" ]] || continue
   run cp -a --reflink=auto "$path/$d" "$STUDY_ROOT/$d"
  done
  echo 'Historical snapshot copied only in --execute mode. Absolute paths and source hashes remain unchanged and must pass original checks.'
  ;;
 *) echo 'Usage: load_data.sh [--execute] {csaw|inbreast|cbis|artifacts} PATH' >&2; exit 2 ;;
esac
