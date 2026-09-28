#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
case "${1:-}" in
 test)
  run "$PYTHON" scripts/comparative_held32/prepare.py
  run "$PYTHON" scripts/comparative_held32/infer.py --gpu "${GPU_INDEX:-0}"
  run "$PYTHON" scripts/comparative_held32/evaluate.py
  ;;
 deepmica|yolo) run "$PYTHON" scripts/additional_baselines/training/internal.py --model "$1" --device "${DEVICE:-cuda:1}" ;;
 external) run "$PYTHON" scripts/external_current_models/run_queue.py --device "${DEVICE:-cuda:1}" ;;
 *) echo 'Usage: evaluate.sh [--execute] {test|deepmica|yolo|external}'; exit 2 ;;
esac
