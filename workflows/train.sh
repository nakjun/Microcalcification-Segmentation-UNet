#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
model="${1:-}"; stage="${2:-}"; gpu="${GPU_INDEX:-0}"
case "$model" in
 proposed|unet2d|utnet)
  export STUDY_BACKBONE="$model"; code=scripts/comparative_study; output="runs/comparative_study/${model}_seed42"; report="reports/comparative_study/${model}_seed42" ;;
 unext) code=scripts/improvement_unext_deterministic; output=runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42; report=reports/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42 ;;
 deepmica|yolo)
  [[ "$stage" == train ]] || { echo 'Additional models use stage train (original worker includes internal evaluation)'; exit 2; }
  require_file runs/additional_baselines_20260926/protocol.json
  require_file "runs/additional_baselines_20260926/preflight/$model.json"
  require_file runs/additional_baselines_20260926/data/prepared_complete.json
  require_file .venv/bin/python
  # Original worker fixes cuda:1, seed42, epoch16, final checkpoint, and serializes via GPU lock.
  run "$PYTHON" scripts/additional_baselines/training/worker.py --model "$model"
  exit ;;
 *) echo 'Usage: train.sh [--execute] {proposed|unet2d|utnet|unext|deepmica|yolo} {prepare|smoke|warmup|train|infer|evaluate}'; exit 2 ;;
esac
require_file runs/calcification_refinement_v5_seed42/complete.json
case "$stage" in
 prepare) run "$PYTHON" "$code/train.py" --mode prepare ;;
 smoke)
  run "$PYTHON" "$code/verify_training.py" --mode cpu
  for variant in C L H LH; do
   for fold in 0 1 2; do
    run "$PYTHON" "$code/train.py" --mode smoke --variant "$variant" --fold "$fold" --gpu "$gpu"
   done
  done
  run "$PYTHON" "$code/verify_training.py" --mode smoke
  run "$PYTHON" "$code/verify_gpu_native.py" --gpu "$gpu"
  ;;
 warmup|train|infer)
  if [[ "$stage" != infer ]]; then require_file "$report/preflight.json"; fi
  if [[ "$stage" == train ]]; then require_file "$output/warmup_verified.json"; fi
  for variant in C L H LH; do
   for fold in 0 1 2; do
    run "$PYTHON" "$code/train.py" --mode "$stage" --variant "$variant" --fold "$fold" --gpu "$gpu"
   done
  done
  if [[ "$stage" == warmup ]]; then run "$PYTHON" "$code/verify_warmup.py"; fi
  ;;
 evaluate) run "$PYTHON" "$code/evaluate.py" ;;
 *) echo 'Unknown stage'; exit 2 ;;
esac
