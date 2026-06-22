#!/bin/bash

# LLaDA KLASS+TACG on MBPP
GPU_ID=${GPU_ID:-5}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/LLaDa-8B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/llada_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --dataset mbpp \
  --gen_length 256 \
  --block_length 64 \
  --steps 256 \
  --alg confidence_threshold_tilg_history_gate_capped_extra \
  --unmask_strategy all \
  --conf_threshold 0.7 \
  --kl_threshold 0.01 \
  --history_length 2 \
  --guidance_weight 0.40 \
  --tilg_ema_decay 0.95 \
  --tilg_rerank_lambda 0.12 \
  --tilg_extra_conf_floor 0.60 \
  --tilg_extra_ratio 0.25 \
  --tilg_extra_max 1 \
  --tilg_extra_allow_empty_base 0 \
  --history_gate_min_streak 2 \
  --history_gate_confidence_escape 0.93 \
  --save_steps