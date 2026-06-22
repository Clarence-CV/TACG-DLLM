#!/bin/bash

# LLaDA KLASS+TACG on GSM8K
GPU_ID=${GPU_ID:-5}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/LLaDa-8B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/llada_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --dataset gsm8k \
  --gen_length 256 \
  --block_length 64 \
  --steps 256 \
  --alg confidence_threshold_tilg_history_gate_capped_extra \
  --unmask_strategy all \
  --conf_threshold 0.65 \
  --kl_threshold 0.015 \
  --history_length 2 \
  --guidance_weight 0.3 \
  --tilg_ema_decay 0.95 \
  --tilg_rerank_lambda 0.03 \
  --tilg_extra_conf_floor 0.60 \
  --tilg_extra_max 1 \
  --tilg_extra_allow_empty_base 0 \
  --history_gate_min_streak 2 \
  --save_steps