#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-src}"
PYTHON_BIN="${PYTHON_BIN:-/data01/anaconda/envs/MMIL_env/bin/python}"

DATASET="summe"
SPLIT_FILE="${SPLIT_FILE:-splits/summe.yml}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-19500}"
MAX_EPOCH="${MAX_EPOCH:-100}"
MAX_SPLITS="${MAX_SPLITS:-}"

LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-5}"

LAMBDA_PAIR="${LAMBDA_PAIR:-0.2}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-1.0}"
LAMBDA_AUX="${LAMBDA_AUX:-2.0}"

RANK_LOSS="${RANK_LOSS:-budgeted_pseudo_summary}"
SCORE_HEAD="${SCORE_HEAD:-dual}"
SELECTION_SCORE_SOURCE="${SELECTION_SCORE_SOURCE:-frame}"
SHOT_HEAD_MODE="${SHOT_HEAD_MODE:-single}"
SHOT_EVAL_HEAD="${SHOT_EVAL_HEAD:-selection}"
UTILITY_FORMULA="${UTILITY_FORMULA:-phase1_default}"

SHOT_UTILITY_PATH="${SHOT_UTILITY_PATH:-}"
TEXT_FEATURE_PATH="${TEXT_FEATURE_PATH:-}"
STRUCTURED_CAPTION_PATH="${STRUCTURED_CAPTION_PATH:-}"
PREFERENCE_TEACHER_PATH="${PREFERENCE_TEACHER_PATH:-}"

LAMBDA_PREF_PAIR="${LAMBDA_PREF_PAIR:-0.2}"
LAMBDA_PREF_LIST="${LAMBDA_PREF_LIST:-0.1}"
LAMBDA_PREF_INCLUSION="${LAMBDA_PREF_INCLUSION:-0.05}"
LAMBDA_PREF_BUDGET="${LAMBDA_PREF_BUDGET:-0.02}"
PREF_CONFIDENCE_THRESHOLD="${PREF_CONFIDENCE_THRESHOLD:-0.6}"
PREF_PAIR_MARGIN="${PREF_PAIR_MARGIN:-0.05}"
PREF_LIST_TEMPERATURE="${PREF_LIST_TEMPERATURE:-0.2}"

LAMBDA_LISTWISE="${LAMBDA_LISTWISE:-0.2}"
LISTWISE_TEMPERATURE="${LISTWISE_TEMPERATURE:-0.2}"
LAMBDA_SELECT="${LAMBDA_SELECT:-0.2}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0.05}"
USE_DIFF_BUDGET_SELECTOR="${USE_DIFF_BUDGET_SELECTOR:-0}"
DIFF_BUDGET_TAU="${DIFF_BUDGET_TAU:-0.2}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0.01}"
SUMMARY_BUDGET="${SUMMARY_BUDGET:-0.15}"
NEGATIVE_QUANTILE="${NEGATIVE_QUANTILE:-0.25}"
TEACHER_GATE_MODE="${TEACHER_GATE_MODE:-none}"
TEACHER_MARGIN_THRESHOLD="${TEACHER_MARGIN_THRESHOLD:-0.0}"

TEXT_COND_NUM="${TEXT_COND_NUM:-10}"
CAPTION_COVERAGE_AWARE="${CAPTION_COVERAGE_AWARE:-1}"
COVERAGE_LOSS_MIN_WEIGHT="${COVERAGE_LOSS_MIN_WEIGHT:-0.5}"

BASE_MODEL="${BASE_MODEL:-attention}"
NUM_HEAD="${NUM_HEAD:-8}"
NUM_FEATURE="${NUM_FEATURE:-768}"
NUM_HIDDEN="${NUM_HIDDEN:-128}"

RUN_ROOT="${RUN_ROOT:-models/mil_cond}"
RUN_TAG="${RUN_TAG:-summe_${UTILITY_FORMULA}_seed${SEED}}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-log_mil_cond.txt}"

mkdir -p "${MODEL_DIR}"

EXTRA_ARGS=()
if [[ "${CAPTION_COVERAGE_AWARE}" == "1" ]]; then
  EXTRA_ARGS+=("--caption-coverage-aware")
fi
if [[ -n "${MAX_SPLITS}" ]]; then
  EXTRA_ARGS+=("--max-splits" "${MAX_SPLITS}")
fi
if [[ "${USE_DIFF_BUDGET_SELECTOR}" == "1" ]]; then
  EXTRA_ARGS+=("--use-diff-budget-selector")
fi
if [[ -n "${SHOT_UTILITY_PATH}" ]]; then
  EXTRA_ARGS+=("--shot-utility-path" "${SHOT_UTILITY_PATH}")
fi
if [[ -n "${TEXT_FEATURE_PATH}" ]]; then
  EXTRA_ARGS+=("--text-feature-path" "${TEXT_FEATURE_PATH}")
fi
if [[ -n "${STRUCTURED_CAPTION_PATH}" ]]; then
  EXTRA_ARGS+=("--structured-caption-path" "${STRUCTURED_CAPTION_PATH}")
fi
if [[ -n "${PREFERENCE_TEACHER_PATH}" ]]; then
  EXTRA_ARGS+=("--preference-teacher-path" "${PREFERENCE_TEACHER_PATH}")
fi

PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" src/run_train_mil_cond.py \
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
  --rank-loss "${RANK_LOSS}" \
  --score-head "${SCORE_HEAD}" \
  --selection-score-source "${SELECTION_SCORE_SOURCE}" \
  --shot-head-mode "${SHOT_HEAD_MODE}" \
  --shot-eval-head "${SHOT_EVAL_HEAD}" \
  --utility-formula "${UTILITY_FORMULA}" \
  --lambda-listwise "${LAMBDA_LISTWISE}" \
  --listwise-temperature "${LISTWISE_TEMPERATURE}" \
  --lambda-select "${LAMBDA_SELECT}" \
  --lambda-budget "${LAMBDA_BUDGET}" \
  --diff-budget-tau "${DIFF_BUDGET_TAU}" \
  --lambda-delta "${LAMBDA_DELTA}" \
  --lambda-pref-pair "${LAMBDA_PREF_PAIR}" \
  --lambda-pref-list "${LAMBDA_PREF_LIST}" \
  --lambda-pref-inclusion "${LAMBDA_PREF_INCLUSION}" \
  --lambda-pref-budget "${LAMBDA_PREF_BUDGET}" \
  --pref-confidence-threshold "${PREF_CONFIDENCE_THRESHOLD}" \
  --pref-pair-margin "${PREF_PAIR_MARGIN}" \
  --pref-list-temperature "${PREF_LIST_TEMPERATURE}" \
  --summary-budget "${SUMMARY_BUDGET}" \
  --negative-quantile "${NEGATIVE_QUANTILE}" \
  --teacher-gate-mode "${TEACHER_GATE_MODE}" \
  --teacher-margin-threshold "${TEACHER_MARGIN_THRESHOLD}" \
  --text-cond-num "${TEXT_COND_NUM}" \
  --coverage-loss-min-weight "${COVERAGE_LOSS_MIN_WEIGHT}" \
  --base-model "${BASE_MODEL}" \
  --num-head "${NUM_HEAD}" \
  --num-feature "${NUM_FEATURE}" \
  --num-hidden "${NUM_HIDDEN}" \
  "${EXTRA_ARGS[@]}"
