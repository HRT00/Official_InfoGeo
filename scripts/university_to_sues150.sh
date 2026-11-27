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
  --source university --train-root "$data/University-1652" \
  --epochs 40 --first-eval 3 \
  --batch-size 4 \
  --lr 0.000635 --warmup-epochs 0.1 \
  --rec 0.05 --dis 0.85 --reg 0.05 \
  --university-drone-view-range 41 49 --cacs-weight "${CACS_WEIGHT:-0.5}" \
  --backbone-lr-scale 0.7 --stable-graph-gradients \
  --output "$repo/logs/university_to_sues150" "$@"
