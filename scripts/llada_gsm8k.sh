#!/bin/bash

GPU_ID=${GPU_ID:-5}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/LLaDa-8B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

GEN_LENGTH=256
BLOCK_LENGTH=64
STEPS=256

DATASET="gsm8k"

# LLaDA confidence-style setting (KL effectively disabled)
ALG="klass"
UNMASK_STRATEGY="all"

CONF_THRESHOLD=0.9
KL_THRESHOLD=100000
HISTORY_LENGTH=1

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/llada_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --gen_length $GEN_LENGTH \
  --block_length $BLOCK_LENGTH \
  --steps $STEPS \
  --conf_threshold $CONF_THRESHOLD \
  --kl_threshold $KL_THRESHOLD \
  --history_length $HISTORY_LENGTH \
  --dataset $DATASET \
  --alg $ALG \
  --unmask_strategy $UNMASK_STRATEGY \
  --save_steps