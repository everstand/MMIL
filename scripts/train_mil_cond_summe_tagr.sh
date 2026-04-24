#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-src}"

DATASET="summe"
SPLIT_FILE="${SPLIT_FILE:-splits/summe.yml}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-19500}"
MAX_EPOCH="${MAX_EPOCH:-100}"

LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-5}"

LAMBDA_PAIR="${LAMBDA_PAIR:-0.2}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-1.0}"
LAMBDA_AUX="${LAMBDA_AUX:-2.0}"

TEXT_COND_NUM="${TEXT_COND_NUM:-10}"

BASE_MODEL="${BASE_MODEL:-attention}"
NUM_HEAD="${NUM_HEAD:-8}"
NUM_FEATURE="${NUM_FEATURE:-768}"
NUM_HIDDEN="${NUM_HIDDEN:-128}"

RUN_ROOT="${RUN_ROOT:-models/mil_cond}"
RUN_TAG="${RUN_TAG:-summe25_spair_attention_lr${LR}_wd${WEIGHT_DECAY}_lp${LAMBDA_PAIR}_pm${PAIR_MARGIN}_la${LAMBDA_ALIGN}_laux${LAMBDA_AUX}_m${TEXT_COND_NUM}_seed${SEED}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-log_mil_cond.txt}"

mkdir -p "${MODEL_DIR}"

PYTHONPATH="${PYTHONPATH}" python src/run_train_mil_cond.py \
  --dataset "${DATASET}" \
  --splits "${SPLIT_FILE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --max-epoch "${MAX_EPOCH}" \
  --model-dir "${MODEL_DIR}" \
  --log-file "${LOG_FILE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lambda-pair "${LAMBDA_PAIR}" \
  --pair-margin "${PAIR_MARGIN}" \
  --lambda-align "${LAMBDA_ALIGN}" \
  --lambda-aux "${LAMBDA_AUX}" \
  --text-cond-num "${TEXT_COND_NUM}" \
  --base-model "${BASE_MODEL}" \
  --num-head "${NUM_HEAD}" \
  --num-feature "${NUM_FEATURE}" \
  --num-hidden "${NUM_HIDDEN}"
