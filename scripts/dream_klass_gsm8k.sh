#!/bin/bash

# Dream KLASS + TILG (GSM8K best)
GPU_ID=${GPU_ID:-4}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/Dream-v0-Instruct-7B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

GEN_LENGTH=256
STEPS=256
DATASET="gsm8k"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/dream_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --dataset $DATASET \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --alg dream_klass_tilg \
  --unmask_strategy all \
  --conf_threshold 0.9 \
  --kl_threshold 0.001 \
  --history_length 2 \
  --guidance_weight 0.3 \
  --tilg_ema_decay 0.95 \
  --tilg_rerank_lambda 0.009 \
  --tilg_extra_conf_floor 0.79 \
  --tilg_extra_ratio 0.04 \
  --tilg_extra_max 1 \
  --tilg_extra_allow_empty_base 0 \
  --history_gate_min_streak 4 \
  --history_gate_confidence_escape 0.982 \
  --save_steps
