#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data="${DATA_ROOT:-$repo/data}"
mode="${1:-train}"
if (( $# )); then shift; fi
evaluation=(
  --target sues --test-root "$data/SUES-200/SUES-200-512x512/Testing/150"
  --eval-batch-size 128 --num-workers 7 --seed 1 --num-slots 16 --alpha 0.8
)
case "$mode" in
  eval) exec "${PYTHON:-python}" "$repo/eval_cvgl_slot.py" "${evaluation[@]}" "$@" ;;
  train) ;;
  *) echo "Usage: $0 {train|eval} [options]" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" "$repo/train_reference.py" "${evaluation[@]}" \
  --source dense --train-root "$data/DenseUAV" \
  --epochs 1 --first-eval 1 \
  --batch-size 4 \
  --lr 0.00065 --warmup-epochs 0.15 \
  --rec 0.05 --dis 0.85 --reg 0.05 \
  --dense-drone-height 80 --square-crop-dense-drone --cacs-weight 1 \
  --output "$repo/logs/dense_to_sues150" "$@"
