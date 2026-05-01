#!/bin/bash

# KD Student Model Training Script for Transfuser++
# Trains a student model under knowledge distillation from a pretrained teacher

set -e  # Exit on any error

# Resolve paths relative to this script so it can be run from this directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)/this_studio"

# Configuration
PYTHON_BIN="${PYTHON_BIN:-/home/haemanth/miniconda3/envs/tf++/bin/python}"
CONFIG_PATH="$REPO_ROOT/models/pretrained_models/all_towns/config.json"
TEACHER_CKPT="$REPO_ROOT/models/pretrained_models/all_towns/model_final_merged.pth"
DATA_ROOT="/teamspace/studios/this_studio/idd_processed"
OUTPUT_DIR="$REPO_ROOT/output"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-student_kd_experiment}"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"

# ── CHECKPOINT & MODE SELECTION ──
# Set LOADFILE and CONTINUE_EPOCH to control training mode:
#   1. FINE-TUNE MODE (continue_epoch=0):
#      - LOADFILE=path/to/model.pth, CONTINUE_EPOCH=0
#      - Loads only model weights from LOADFILE, resets optimizer/scheduler/scaler
#      - Starts from epoch 0
#      - LOADFILE can be omitted for fresh training from random init
#   2. RESUME MODE (continue_epoch>0):
#      - LOADFILE=path/to/model_000x.pth, CONTINUE_EPOCH=1 (or any >0)
#      - Loads full checkpoint state (model + optimizer + scheduler + scaler + metadata)
#      - Validates detect_boxes matches saved metadata
#      - Continues from saved epoch
#      - LOADFILE is REQUIRED in this mode
LOADFILE="${LOADFILE:-/teamspace/studios/this_studio/student_model/student_best.pth}"
CONTINUE_EPOCH="${CONTINUE_EPOCH:-0}"

# ── BBOX-ONLY TRAINING ──
# Set BBOX_ONLY_TRAIN=1 to train only the backbone + bbox head.
# In this mode, planning head is skipped, and only bbox labels are required.
BBOX_ONLY_TRAIN="${BBOX_ONLY_TRAIN:-1}"

# Training hyperparameters
EPOCHS=2
BATCH_SIZE=8
LR=1e-9
WEIGHT_DECAY=1e-4
NUM_WORKERS=2
SAVE_EVERY=5
LOG_EVERY=1000

# Loss weights for KD
W_CK=1.0          # Checkpoint loss weight
W_TS=1.0          # Target speed loss weight
W_FEAT=0.5        # Feature KD loss weight
W_KD=0.5          # Output KD loss weight
W_BBOX=1.0        # Bounding box loss weight
TEMPERATURE=4.0   # KD temperature

echo "========================================="
echo "KD Training Configuration"
echo "========================================="
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Teacher: $TEACHER_CKPT"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Run Output Dir: $RUN_OUTPUT_DIR"
echo "Loadfile: ${LOADFILE:-<none>}"
echo "Continue Epoch: $CONTINUE_EPOCH"
echo "BBox-only Train: $BBOX_ONLY_TRAIN"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Num Workers: $NUM_WORKERS"
echo "========================================="

# # Basic sanity checks before starting.
# if [ ! -x "$PYTHON_BIN" ]; then
#     echo "Error: Python executable not found or not executable: $PYTHON_BIN"
#     exit 1
# fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Missing config: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Error: Missing teacher checkpoint: $TEACHER_CKPT"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Missing data root: $DATA_ROOT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$RUN_OUTPUT_DIR"

# Run training
cd "$REPO_ROOT"

TRAIN_ARGS=(
    --config_path "$CONFIG_PATH"
    --teacher_ckpt "$TEACHER_CKPT"
    --data_root "$DATA_ROOT"
    --output_dir "$OUTPUT_DIR"
    --experiment_name "$EXPERIMENT_NAME"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --num_workers "$NUM_WORKERS"
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
    --w_ck "$W_CK"
    --w_ts "$W_TS"
    --w_feat "$W_FEAT"
    --w_kd "$W_KD"
    --w_bbox "$W_BBOX"
    --temperature "$TEMPERATURE"
)

if [ -n "$LOADFILE" ]; then
    TRAIN_ARGS+=(--loadfile "$LOADFILE")
    TRAIN_ARGS+=(--continue_epoch "$CONTINUE_EPOCH")
fi

if [ "$BBOX_ONLY_TRAIN" = "1" ]; then
    TRAIN_ARGS+=(--bbox_only_train)
fi

python -m model_nocarla.train \
    "${TRAIN_ARGS[@]}"

echo "========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================="
