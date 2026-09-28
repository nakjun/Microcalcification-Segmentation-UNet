#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
session="${1:?Usage: tmux_run.sh SESSION {train|evaluate|preprocess} ARGUMENTS...}"
workflow="${2:?Missing workflow}"; shift 2
[[ "$session" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid session name'; exit 2; }
case "$workflow" in train|evaluate|preprocess) ;; *) echo 'Invalid workflow'; exit 2;; esac
command -v tmux >/dev/null
printf -v command 'cd %q && bash %q --execute' "$root" "$root/workflows/$workflow.sh"
for arg in "$@"; do printf -v quoted ' %q' "$arg"; command+="$quoted"; done
# Duplicate session names fail; no existing session is replaced.
tmux new-session -d -s "$session" "$command"
echo "Started session: $session (attach: tmux attach -t $session)"
