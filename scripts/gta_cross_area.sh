#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data="${DATA_ROOT:-$repo/data}"
mode="${1:-train}"
if (( $# )); then shift; fi
evaluation=(
  --target gta --test-root "$data/GTA-UAV"
  --eval-batch-size 128 --num-workers 7 --seed 1 --num-slots 16 --alpha 0.8
)
case "$mode" in
  eval) exec "${PYTHON:-python}" "$repo/eval_cvgl_slot.py" "${evaluation[@]}" "$@" ;;
  train) ;;
  *) echo "Usage: $0 {train|eval} [options]" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" "$repo/train_reference.py" "${evaluation[@]}" \
  --source gta --train-root "$data/GTA-UAV" \
  --epochs 7 --first-eval 2 \
  --batch-size 16 \
  --lr 0.000685 --warmup-epochs 0.15 \
  --rec 0.05 --dis 0.85 --reg 0.05 \
  --output "$repo/logs/gta_cross_area" "$@"
