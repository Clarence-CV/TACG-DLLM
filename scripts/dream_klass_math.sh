#!/bin/bash

# Dream KLASS + TILG (MATH best)
GPU_ID=5
MODEL_PATH="/data/ckpt/Dream-v0-Instruct-7B"
SAVE_DIR="./results_final"

GEN_LENGTH=256
STEPS=256
DATASET="math"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./src/dream_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --dataset $DATASET \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --alg dream_klass_tilg \
  --unmask_strategy all \
  --conf_threshold 0.8 \
  --kl_threshold 0.001 \
  --history_length 2 \
  --guidance_weight 0.3 \
  --tilg_ema_decay 0.95 \
  --tilg_rerank_lambda 0.01 \
  --tilg_extra_conf_floor 0.75 \
  --tilg_extra_ratio 0.05 \
  --tilg_extra_max 1 \
  --tilg_extra_allow_empty_base 1 \
  --history_gate_min_streak 3 \
  --history_gate_confidence_escape 0.98 \
  --save_steps