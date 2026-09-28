#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"
PYTHON="${PYTHON:-$STUDY_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
EXECUTE=0
if [[ "${1:-}" == --execute ]]; then EXECUTE=1; shift; fi
run() {
  printf '%q ' "$@"; printf '\n'
  if (( EXECUTE )); then "$@"; fi
}
require_file() {
  if (( EXECUTE )) && [[ ! -f "$1" ]]; then
    printf 'Missing required original input: %s\nSee workflows/README.md.\n' "$1" >&2; exit 2
  fi
}
