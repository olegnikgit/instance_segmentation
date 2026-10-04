#!/usr/bin/env bash

set -euo pipefail

# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_SCRIPT="src/instance_segmentation/script/rf_detr/rf_detr_instance_segmentation_train.py"

DATASET_ROOT="${HOME}/Documents/Databases/Beams_instances/rf_detr_5fold"

OUTPUT_ROOT="${HOME}/Documents/experiments/Beams_instances/rf_detr/rf_detr_5fold"

NUM_FOLDS=5

EPOCHS=200
BATCH_SIZE=2
RESOLUTION=624
NUM_WORKERS=4
LR=1e-4
SEED=42

# ==========================================================
# CHECKS
# ==========================================================

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found:"
    echo "  $TRAIN_SCRIPT"
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: Dataset root not found:"
    echo "  $DATASET_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo
echo "=========================================================="
echo "RF-DETR ${NUM_FOLDS}-FOLD CROSS-VALIDATION"
echo "=========================================================="
echo "Dataset root : $DATASET_ROOT"
echo "Output root  : $OUTPUT_ROOT"
echo "Model        : RFDETRSegSmall"
echo "Resolution   : $RESOLUTION"
echo "Epochs       : $EPOCHS"
echo "Batch size   : $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Seed         : $SEED"
echo "=========================================================="
echo

# ==========================================================
# RUN FOLDS
# ==========================================================

for ((FOLD=1; FOLD<=NUM_FOLDS; FOLD++)); do

    FOLD_NAME=$(printf "fold_%02d" "$FOLD")

    DATASET_DIR="${DATASET_ROOT}/${FOLD_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${FOLD_NAME}"

    if [[ ! -d "$DATASET_DIR" ]]; then
        echo
        echo "ERROR: Dataset for ${FOLD_NAME} not found:"
        echo "  $DATASET_DIR"
        exit 1
    fi

    echo
    echo "=========================================================="
    echo "STARTING ${FOLD_NAME}"
    echo "=========================================================="
    echo "Dataset: $DATASET_DIR"
    echo "Output : $OUTPUT_DIR"
    echo

    mkdir -p "$OUTPUT_DIR"

    python -u "$TRAIN_SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --resolution "$RESOLUTION" \
        --num_workers "$NUM_WORKERS" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "${OUTPUT_DIR}/training.log"

    echo
    echo "=========================================================="
    echo "${FOLD_NAME} FINISHED"
    echo "=========================================================="

    BEST_MODEL="${OUTPUT_DIR}/checkpoint_best_total.pth"

    if [[ ! -f "$BEST_MODEL" ]]; then
        echo "WARNING: Best checkpoint not found:"
        echo "  $BEST_MODEL"
    else
        echo "Best checkpoint:"
        echo "  $BEST_MODEL"
    fi

done

# ==========================================================
# FINAL SUMMARY
# ==========================================================

echo
echo "=========================================================="
echo "ALL FOLDS FINISHED"
echo "=========================================================="

for ((FOLD=1; FOLD<=NUM_FOLDS; FOLD++)); do

    FOLD_NAME=$(printf "fold_%02d" "$FOLD")
    OUTPUT_DIR="${OUTPUT_ROOT}/${FOLD_NAME}"

    echo
    echo "${FOLD_NAME}:"

    if [[ -f "${OUTPUT_DIR}/checkpoint_best_total.pth" ]]; then
        echo "  checkpoint: OK"
    else
        echo "  checkpoint: MISSING"
    fi

    if [[ -f "${OUTPUT_DIR}/training.log" ]]; then
        echo "  log       : ${OUTPUT_DIR}/training.log"
    fi

done

echo
echo "Results are stored in:"
echo "  $OUTPUT_ROOT"


