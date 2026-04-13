#!/usr/bin/env bash
set -euo pipefail

DATASET="tvsum"
SPLIT_FILE="${SPLIT_FILE:-splits/tvsum.yml}"
MODEL_DIR="${MODEL_DIR:-models/mil/tvsum_run}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-12345}"
MAX_EPOCH="${MAX_EPOCH:-300}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BASE_MODEL="${BASE_MODEL:-attention}"
NUM_HEAD="${NUM_HEAD:-8}"
NUM_FEATURE="${NUM_FEATURE:-1024}"
NUM_HIDDEN="${NUM_HIDDEN:-128}"

python src/run_train_mil.py \
  --dataset "${DATASET}" \
  --splits "${SPLIT_FILE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --max-epoch "${MAX_EPOCH}" \
  --model-dir "${MODEL_DIR}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --base-model "${BASE_MODEL}" \
  --num-head "${NUM_HEAD}" \
  --num-feature "${NUM_FEATURE}" \
  --num-hidden "${NUM_HIDDEN}"