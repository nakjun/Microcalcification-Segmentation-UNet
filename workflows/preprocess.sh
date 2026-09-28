#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
case "${1:-}" in
 native)
  require_file configs/calcification_refinement.json
  run "$PYTHON" scripts/prepare_calcification_refinement.py --config configs/calcification_refinement.json
  ;;
 baseline) run "$PYTHON" scripts/prepare_calcification_refinement_baselines.py ;;
 additional)
  require_file runs/calcification_refinement_v5_seed42/folds.json
  run "$PYTHON" scripts/additional_baselines/training/prepare.py --output runs/additional_baselines_20260926/data
  ;;
 test) run "$PYTHON" scripts/comparative_held32/prepare.py ;;
 external) run "$PYTHON" scripts/external_current_models/infer.py --prepare ;;
 *) echo 'Usage: preprocess.sh [--execute] {native|baseline|additional|test|external}'; exit 2 ;;
esac
