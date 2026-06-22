#!/bin/bash

# Dream KLASS baseline + TILG (conservative)
GPU_ID=${GPU_ID:-4}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/Dream-v0-Instruct-7B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

GEN_LENGTH=256
STEPS=256
DATASET="humaneval"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/dream_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --dataset $DATASET \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --alg klass_ours_full \
  --unmask_strategy all \
  --conf_threshold 0.8 \
  --kl_threshold 0.001 \
  --history_length 2 \
  --guidance_weight 0.3 \
  --tilg_ema_decay 0.95 \
  --tilg_rerank_lambda 0.01 \
  --tilg_extra_conf_floor 0.7 \
  --tilg_extra_kl_threshold 0.001 \
  --history_gate_min_streak 3 \
  --history_gate_confidence_escape 0.97 \
  --save_steps