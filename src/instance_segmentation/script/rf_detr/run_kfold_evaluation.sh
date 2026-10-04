#!/usr/bin/env bash

set -euo pipefail

# ==========================================================
# PATHS
# ==========================================================

SCRIPT_DIR="src/instance_segmentation/script/rf_detr"
PYTHON="python"

# ==========================================================
# DATA / EXPERIMENT PATHS
# ==========================================================

DATASET_ROOT="${HOME}/Documents/Databases/Beams_instances/rf_detr_5fold"
CV_RESULTS_ROOT="${HOME}/Documents/experiments/Beams_instances/rf_detr/rf_detr_5fold"

INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_kfold.py"
EVALUATION_SCRIPT="${SCRIPT_DIR}/evaluate_kfold.py"

# ==========================================================
# SETTINGS
# ==========================================================

NUM_FOLDS=5

CONFIDENCE=0.5
IOU_THRESHOLD=0.5


mkdir -p "$CV_RESULTS_ROOT"

echo
echo "=========================================================="
echo "RF-DETR K-FOLD INFERENCE + EVALUATION"
echo "=========================================================="
echo "Dataset root : $DATASET_ROOT"
echo "Results root : $CV_RESULTS_ROOT"
echo "Model        : checkpoint_best_total.pth"
echo "Confidence   : $CONFIDENCE"
echo "BBox IoU     : $IOU_THRESHOLD"
echo "Folds        : $NUM_FOLDS"
echo "=========================================================="
echo

# ==========================================================
# INFERENCE FOR EACH FOLD
# ==========================================================

for ((FOLD=1; FOLD<=NUM_FOLDS; FOLD++)); do

    FOLD_NAME=$(printf "fold_%02d" "$FOLD")

    FOLD_DIR="${CV_RESULTS_ROOT}/${FOLD_NAME}"
    DATASET_FOLD="${DATASET_ROOT}/${FOLD_NAME}"

    TRAIN_DIR="${DATASET_FOLD}/train"
    VALID_DIR="${DATASET_FOLD}/valid"

    MODEL="${FOLD_DIR}/checkpoint_best_total.pth"

    PREDICTION_DIR="${FOLD_DIR}/predictions"
    PREDICTION_JSON="${PREDICTION_DIR}/predictions.json"

    if [[ ! -f "$MODEL" ]]; then
        echo
        echo "ERROR: Model checkpoint not found:"
        echo "  $MODEL"
        exit 1
    fi

    if [[ ! -d "$TRAIN_DIR" ]]; then
        echo
        echo "ERROR: Training directory not found:"
        echo "  $TRAIN_DIR"
        exit 1
    fi

    if [[ ! -d "$VALID_DIR" ]]; then
        echo
        echo "ERROR: Validation directory not found:"
        echo "  $VALID_DIR"
        exit 1
    fi

    mkdir -p "$PREDICTION_DIR"

    echo
    echo "=========================================================="
    echo "INFERENCE: ${FOLD_NAME}"
    echo "=========================================================="
    echo "Model   : $MODEL"
    echo "Context : $TRAIN_DIR"
    echo "Query   : $VALID_DIR"
    echo "Output  : $PREDICTION_JSON"
    echo

    "$PYTHON" -u "$INFERENCE_SCRIPT" \
        --model_weights "$MODEL" \
        --context_dir "$TRAIN_DIR" \
        --predict_dir "$VALID_DIR" \
        --output_json "$PREDICTION_JSON" \
        --confidence "$CONFIDENCE" \
        --save_masks \
        --save_overlay

    echo
    echo "${FOLD_NAME} inference finished."

done

# ==========================================================
# EVALUATION
# ==========================================================

echo
echo "=========================================================="
echo "EVALUATING ALL FOLDS"
echo "=========================================================="

"$PYTHON" -u "$EVALUATION_SCRIPT" \
    --cv_dir "$CV_RESULTS_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --prediction_name "predictions.json" \
    --confidence "$CONFIDENCE" \
    --iou_threshold "$IOU_THRESHOLD"

echo
echo "=========================================================="
echo "K-FOLD EVALUATION FINISHED"
echo "=========================================================="

echo
echo "Results:"
echo "  $CV_RESULTS_ROOT/evaluation/fold_metrics.csv"
echo "  $CV_RESULTS_ROOT/evaluation/cv_summary.json"
