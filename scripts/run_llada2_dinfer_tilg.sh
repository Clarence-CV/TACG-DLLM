#!/bin/bash
set -euo pipefail

# Run from anywhere; this script is designed to live in TILg_final/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== Modes =====
# MODE=baseline -> LLaDA2 origin decode baseline
# MODE=tilg     -> LLaDA2 origin decode + HG/TILG args
MODE="${MODE:-tilg}"

# ===== Core paths =====
DMAX_ROOT="${DMAX_ROOT:-/home/tzluo/DMax}"
DINFER_EVAL_DIR="${DINFER_EVAL_DIR:-${PROJECT_ROOT}/evaluations}"
DINFER_PYTHONPATH="${DINFER_PYTHONPATH:-${DMAX_ROOT}/dInfer/python}"
MODEL_PATH="${MODEL_PATH:-/data/ckpt/LLaDA2.0-mini}"

# ===== Task setup =====
TASKS="${TASKS:-gsm8k_llada_mini}"
TASKS_INCLUDE_PATH="${TASKS_INCLUDE_PATH:-${PROJECT_ROOT}/tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs_llada2}"

# ===== Runtime =====
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
PARALLEL_DECODING="${PARALLEL_DECODING:-threshold}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"
TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-0}"
GEN_LENGTH="${GEN_LENGTH:-2048}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
THRESHOLD="${THRESHOLD:-0.95}"
LOW_THRESHOLD="${LOW_THRESHOLD:-0.2}"
KL_THRESHOLD="${KL_THRESHOLD:-0.001}"
KL_HISTORY_LENGTH="${KL_HISTORY_LENGTH:-2}"
TILG_EXTRA_KL_THRESHOLD="${TILG_EXTRA_KL_THRESHOLD:-0.001}"
CACHE_MODE="${CACHE_MODE:-prefix}"
WARMUP_TIMES="${WARMUP_TIMES:-0}"
PREFIX_LOOK="${PREFIX_LOOK:-0}"
AFTER_LOOK="${AFTER_LOOK:-0}"
CONT_WEIGHT="${CONT_WEIGHT:-0}"
USE_CREDIT="${USE_CREDIT:-False}"
USE_COMPILE="${USE_COMPILE:-True}"
MODEL_TYPE="${MODEL_TYPE:-llada2}"
USE_BD="${USE_BD:-True}"
MASTER_PORT="${MASTER_PORT:-23456}"
SAVE_SAMPLES="${SAVE_SAMPLES:-True}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# ===== Ours HG + TILG =====
HG_HISTORY_LEN="${HG_HISTORY_LEN:-2}"
HG_MIN_STREAK="${HG_MIN_STREAK:-3}"
TILG_EXTRA_CONF_FLOOR="${TILG_EXTRA_CONF_FLOOR:-0.6}"
TILG_EXTRA_MAX="${TILG_EXTRA_MAX:-1}"
HISTORY_GATE_CONFIDENCE_ESCAPE="${HISTORY_GATE_CONFIDENCE_ESCAPE:-0.95}"
PRESERVE_BASE_THRESHOLD_COMMIT="${PRESERVE_BASE_THRESHOLD_COMMIT:-True}"
TILG_CFG_LIKE_ENABLED="${TILG_CFG_LIKE_ENABLED:-True}"
TILG_GUIDANCE_WEIGHT="${TILG_GUIDANCE_WEIGHT:-0.3}"
TILG_EMA_DECAY="${TILG_EMA_DECAY:-0.95}"
TILG_RERANK_LAMBDA="${TILG_RERANK_LAMBDA:-0.03}"

# ===== Sanity checks =====
if [[ ! -d "${DINFER_EVAL_DIR}" ]]; then
  echo "[ERROR] DINFER_EVAL_DIR not found: ${DINFER_EVAL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DINFER_EVAL_DIR}/eval_dinfer_sglang.py" ]]; then
  echo "[ERROR] eval_dinfer_sglang.py not found under: ${DINFER_EVAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${TASKS_INCLUDE_PATH}" ]]; then
  echo "[ERROR] TASKS_INCLUDE_PATH not found: ${TASKS_INCLUDE_PATH}" >&2
  exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${DINFER_PYTHONPATH}:${PYTHONPATH:-}"

RUN_TAG="${MODE}_$(date +%Y%m%d_%H%M%S)"
FINAL_OUT="${OUTPUT_DIR}/${TASKS}/${RUN_TAG}"
mkdir -p "${FINAL_OUT}"

COMMON_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},low_threshold=${LOW_THRESHOLD},kl_threshold=${KL_THRESHOLD},kl_history_length=${KL_HISTORY_LENGTH},tilg_extra_kl_threshold=${TILG_EXTRA_KL_THRESHOLD},show_speed=True,save_dir=${FINAL_OUT},parallel_decoding=${PARALLEL_DECODING},cache=${CACHE_MODE},warmup_times=${WARMUP_TIMES},use_compile=${USE_COMPILE},tp_size=${TP_SIZE},parallel=${PARALLEL_MODE},cont_weight=${CONT_WEIGHT},use_credit=${USE_CREDIT},prefix_look=${PREFIX_LOOK},after_look=${AFTER_LOOK},gpus='${GPUS}',model_type=${MODEL_TYPE},use_bd=${USE_BD},master_port=${MASTER_PORT},save_samples=${SAVE_SAMPLES}"

if [[ "${MODE}" == "tilg" ]]; then
  MODEL_ARGS="${COMMON_ARGS},hg_history_len=${HG_HISTORY_LEN},hg_min_streak=${HG_MIN_STREAK},tilg_extra_conf_floor=${TILG_EXTRA_CONF_FLOOR},tilg_extra_max=${TILG_EXTRA_MAX},history_gate_confidence_escape=${HISTORY_GATE_CONFIDENCE_ESCAPE},preserve_base_threshold_commit=${PRESERVE_BASE_THRESHOLD_COMMIT},tilg_cfg_like_enabled=${TILG_CFG_LIKE_ENABLED},tilg_guidance_weight=${TILG_GUIDANCE_WEIGHT},tilg_ema_decay=${TILG_EMA_DECAY},tilg_rerank_lambda=${TILG_RERANK_LAMBDA}"
elif [[ "${MODE}" == "baseline" ]]; then
  MODEL_ARGS="${COMMON_ARGS}"
else
  echo "[ERROR] MODE must be 'baseline' or 'tilg', got: ${MODE}" >&2
  exit 1
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] MODE=${MODE}"
echo "[INFO] TASKS=${TASKS}"
echo "[INFO] OUTPUT=${FINAL_OUT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] model_args=${MODEL_ARGS}"
if [[ "${MODE}" == "tilg" ]]; then
  echo "[INFO] TILG params: hg_history_len=${HG_HISTORY_LEN}, hg_min_streak=${HG_MIN_STREAK}, floor=${TILG_EXTRA_CONF_FLOOR}, max=${TILG_EXTRA_MAX}, escape=${HISTORY_GATE_CONFIDENCE_ESCAPE}, cfg_like=${TILG_CFG_LIKE_ENABLED}, gw=${TILG_GUIDANCE_WEIGHT}, ema=${TILG_EMA_DECAY}, rerank=${TILG_RERANK_LAMBDA}, kl=${KL_THRESHOLD}, kl_hist=${KL_HISTORY_LENGTH}, extra_kl=${TILG_EXTRA_KL_THRESHOLD}, preserve_base=${PRESERVE_BASE_THRESHOLD_COMMIT}"
  echo "[INFO] TILG debug toggles: TILG_DEBUG=${TILG_DEBUG:-0}, TILG_ASSERT_EFFECT=${TILG_ASSERT_EFFECT:-0}, TILG_ASSERT_AFTER_STEPS=${TILG_ASSERT_AFTER_STEPS:-50}"
fi

cd "${DINFER_EVAL_DIR}"
python eval_dinfer_sglang.py \
  --tasks "${TASKS}" \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args "${MODEL_ARGS}" \
  --output_path "${FINAL_OUT}" \
  --include_path "${TASKS_INCLUDE_PATH}" \
  --apply_chat_template

echo "[DONE] Results written to: ${FINAL_OUT}"
