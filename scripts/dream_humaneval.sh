#!/bin/bash

GPU_ID=${GPU_ID:-4}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

MODEL_PATH="${MODEL_PATH:-/data/ckpt/Dream-v0-Instruct-7B}"
SAVE_DIR="${SAVE_DIR:-./results_final}"

GEN_LENGTH=256
STEPS=256

DATASET="humaneval"

ALG="confidence_threshold_tilg_history_gate_capped_extra"
UNMASK_STRATEGY="all"

CONF_THRESHOLD=0.9
KL_THRESHOLD=100000
HISTORY_LENGTH=1
GUIDANCE_WEIGHT=0.3
TILG_EMA_DECAY=0.95
TILG_RERANK_LAMBDA=0.05
TILG_EXTRA_CONF_FLOOR=0.8
TILG_EXTRA_RATIO=0.10
TILG_EXTRA_MAX=1
TILG_EXTRA_ALLOW_EMPTY_BASE=0
HISTORY_GATE_MIN_STREAK=3
HISTORY_GATE_CONFIDENCE_ESCAPE=0.97

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./src/dream_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --conf_threshold $CONF_THRESHOLD \
  --kl_threshold $KL_THRESHOLD \
  --history_length $HISTORY_LENGTH \
  --dataset $DATASET \
  --alg $ALG \
  --unmask_strategy $UNMASK_STRATEGY \
  --guidance_weight $GUIDANCE_WEIGHT \
  --tilg_ema_decay $TILG_EMA_DECAY \
  --tilg_rerank_lambda $TILG_RERANK_LAMBDA \
  --tilg_extra_conf_floor $TILG_EXTRA_CONF_FLOOR \
  --tilg_extra_ratio $TILG_EXTRA_RATIO \
  --tilg_extra_max $TILG_EXTRA_MAX \
  --tilg_extra_allow_empty_base $TILG_EXTRA_ALLOW_EMPTY_BASE \
  --history_gate_min_streak $HISTORY_GATE_MIN_STREAK \
  --history_gate_confidence_escape $HISTORY_GATE_CONFIDENCE_ESCAPE \
  --save_steps