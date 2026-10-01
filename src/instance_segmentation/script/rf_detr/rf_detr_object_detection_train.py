#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to train RF-DETR for object detection on custom dataset.


An example of how to use this script:
python rf_detr_object_detection_train.py \
    --dataset_dir <path_to_dataset_directory> \
    --output_dir <path_to_output_directory>


@author: Olegs
"""

# ==========================================================
from rfdetr import RFDETRNano
import argparse
import os


# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # add dataset directory argument
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing images and annotations."
    )

    # add output directory argument
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where the trained model will be saved."
    )

    return parser.parse_args()

def main():

    # parse command line arguments
    args = parse_args()

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # create the model
    model = RFDETRNano(resolution=384)  # 384 is the default resolution for RFDETRNano

    # # try a smaller model for testing purposes
    # model = RFDETRNano(resolution=320,  # 320 is a smaller resolution for testing, original is 384
    #                    num_queries=50)  # reduce the number of queries for testing, original is 300
    #                 #    hidden_dim=128,  # reduce the hidden dimension for testing, original is 256
    #                 #    pretrain_weights=None)  # set to None to train from scratch, original is "<home_directory>/.roboflow/models/rf-detr-nano.pth"
    #                 #                            # we must set pretrain_weights to None because the pre-trained weights are trained on hidden_dim=256, which is incompatible with our smaller model

    # train the model on the custom dataset
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=10,  # number of training epochs
        batch_size=4,  # batch size for training
        lr=1e-4,  # learning rate for training
        num_workers=8,  # number of workers for data loading
        multi_scale = False  # whether to use multi-scale training
    )


# ==========================================================
if __name__ == "__main__":
    main()






