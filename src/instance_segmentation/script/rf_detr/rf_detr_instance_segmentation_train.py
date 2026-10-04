#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train RF-DETR Small for instance segmentation on a custom dataset.

Expected dataset structure:

dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image2.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image3.jpg
    └── ...

Example:

python rf_detr_instance_segmentation_train.py \
    --dataset_dir <path_to_dataset_directory> \
    --output_dir <path_to_output_directory>

@author: Olegs
"""

# ==========================================================
# from rfdetr import RFDETRSegNano
from rfdetr import RFDETRSegSmall

import argparse
import os


# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF-DETR Small for instance segmentation."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the COCO-format dataset directory."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size."
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=624, # 432,
        help="Training image resolution."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training."
    )

    return parser.parse_args()


# ==========================================================
def main():

    args = parse_args()

    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------
    # Create RF-DETR instance segmentation model
    #
    # IMPORTANT:
    # RFDETRSegNano is the segmentation model.
    #
    # Do not use RFDETRNano here.
    # ------------------------------------------------------
    # model = RFDETRSegNano(
    #     resolution=args.resolution,
    #     pretrain_weights="rf-detr-seg-nano.pt"
    # )

    model = RFDETRSegSmall(
        resolution=args.resolution,
        pretrain_weights="rf-detr-seg-small.pt",
    )

    # ------------------------------------------------------
    # Train
    #
    # RF-DETR automatically detects the dataset format
    # from the directory structure.
    #
    # It will also detect the number of classes from
    # the dataset.
    # ------------------------------------------------------
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,

        epochs=args.epochs,
        batch_size=args.batch_size,

        lr=args.lr,

        num_workers=args.num_workers,

        multi_scale=False,

        # Don't run test evaluation during this first
        # training experiment.
        run_test=False,
        seed=args.seed,
        skip_best_epochs=5,  # Skip saving best model for the first 5 epochs.
        # early_stopping=True,  # Enable early stopping based on validation loss.
        # early_stopping_patience=15,  # Stop training if validation loss doesn't improve for 15 epochs.
    )


# ==========================================================
if __name__ == "__main__":
    main()
