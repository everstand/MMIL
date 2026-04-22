#!/usr/bin/env bash
set -euo pipefail

DATASET="summe"

SPLIT_FILE="${SPLIT_FILE:-splits/summe24.yml}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-12345}"
MAX_EPOCH="${MAX_EPOCH:-100}"

LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LAMBDA_AUX="${LAMBDA_AUX:-3.0}"
TEXT_COND_NUM="${TEXT_COND_NUM:-7}"

BASE_MODEL="${BASE_MODEL:-attention}"
NUM_HEAD="${NUM_HEAD:-8}"
NUM_FEATURE="${NUM_FEATURE:-768}"
NUM_HIDDEN="${NUM_HIDDEN:-128}"

RUN_ROOT="${RUN_ROOT:-models/mil_cond}"
RUN_TAG="${RUN_TAG:-${DATASET}_cond_${BASE_MODEL}_lr${LR}_wd${WEIGHT_DECAY}_laux${LAMBDA_AUX}_m${TEXT_COND_NUM}_seed${SEED}}"

MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-log_mil_cond.txt}"

mkdir -p "${MODEL_DIR}"

python src/run_train_mil_cond.py \
  --dataset "${DATASET}" \
  --splits "${SPLIT_FILE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --max-epoch "${MAX_EPOCH}" \
  --model-dir "${MODEL_DIR}" \
  --log-file "${LOG_FILE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lambda-aux "${LAMBDA_AUX}" \
  --text-cond-num "${TEXT_COND_NUM}" \
  --base-model "${BASE_MODEL}" \
  --num-head "${NUM_HEAD}" \
  --num-feature "${NUM_FEATURE}" \
  --num-hidden "${NUM_HIDDEN}"