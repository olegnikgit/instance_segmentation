#!/usr/bin/env bash

set -euo pipefail

# ==========================================================
# USER SETTINGS
# ==========================================================

DB_PATH="/path/to/your/data"

# ==========================================================
# DATASET PREPARATION
# ==========================================================

python src/instance_segmentation/database/prepare_rfdetr_dataset.py \
    --annotations "${DB_PATH}/annotated/annotations.json" \
    --images_dir "${DB_PATH}/annotated/frames" \
    --output_dir "${DB_PATH}/rf_detr_format" \
    --val_count 1 \
    --seed 42

# ==========================================================
# TRAINING
# ==========================================================

python src/instance_segmentation/script/rf_detr/rf_detr_instance_segmentation_train.py \
    --dataset_dir "${DB_PATH}/rf_detr_format" \
    --output_dir "${DB_PATH}/results/train"

# ==========================================================
# INFERENCE
# ==========================================================

python src/instance_segmentation/script/rf_detr/inference.py \
    --model_weights "${DB_PATH}/results/train/checkpoint_best_total.pth" \
    --context_dir "${DB_PATH}/annotated" \
    --predict_dir "${DB_PATH}/unannotated/frames" \
    --output_json "${DB_PATH}/results/test/predictions.json" \
    --confidence 0.5 \
    --save_overlay
