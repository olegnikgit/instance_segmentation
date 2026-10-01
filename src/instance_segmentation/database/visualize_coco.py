#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize COCO annotations.

Functionality:
    * Load the COCO JSON
    * Randomly sample images
    * Draw bounding boxes
    * Display category names
    * Print basic dataset statistics
    * Verify image paths exist
    * Work with your relative-path setup

Example:
python visualize_coco.py \
    --dataset_root <path_to_dataset_root> \
    --coco_json <path_to_coco_json> \
    --num_samples 20

@author: Olegs
"""
# ============================================================
# Imports
# ============================================================
import os
import json
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Arguments
# ============================================================
def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root directory.",
    )

    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Path to COCO annotation file.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to show.",
    )

    return parser.parse_args()


# ============================================================
# COCO Loader
# ============================================================
def load_coco(coco_json):

    with open(coco_json, "r") as f:
        coco = json.load(f)

    return coco


# ============================================================
# Build lookup tables
# ============================================================
def build_indices(coco):
    """
    Build lookup tables for images and categories.
    This allows for efficient access to image info and category names when visualizing annotations.

    Parameters:
    - coco: The COCO dataset loaded from the JSON file, containing "images", "annotations", and "categories" lists.

    Returns:
    - image_lookup: A dictionary mapping image IDs to their corresponding image information (file name, width, height).
    - category_lookup: A dictionary mapping category IDs to their corresponding category names.
    - annotations_per_image: A dictionary mapping image IDs to a list of annotations (bounding boxes, category IDs) for that image.
    """

    # Build lookup tables for images and categories
    image_lookup = {
        img["id"]: img
        for img in coco["images"]
    }
    category_lookup = {
        cat["id"]: cat["name"]
        for cat in coco["categories"]
    }

    # Build a mapping from image_id to its annotations for quick access
    annotations_per_image = {}

    for ann in coco["annotations"]:

        image_id = ann["image_id"]

        annotations_per_image.setdefault(
            image_id,
            []
        ).append(ann)

    return (image_lookup,
            category_lookup,
            annotations_per_image)


# ============================================================
# Statistics
# ============================================================

def print_statistics(coco):
    """
    Print basic statistics about the COCO dataset, including:
    - Number of images
    - Number of annotations
    - Number of categories
    If annotations are present, also print:
    - Minimum bounding box area
    - Mean bounding box area
    - Maximum bounding box area
    """

    num_images = len(coco["images"])
    num_annotations = len(coco["annotations"])
    num_categories = len(coco["categories"])

    print("\n==============================")
    print("COCO DATASET STATISTICS")
    print("==============================")

    print(f"Images      : {num_images}")
    print(f"Annotations : {num_annotations}")
    print(f"Categories  : {num_categories}")

    if num_annotations > 0:
        areas = [ ann["area"] for ann in coco["annotations"] ]
        print( f"Min area    : " f"{np.min(areas):.1f}" )
        print( f"Mean area   : " f"{np.mean(areas):.1f}" )
        print( f"Max area    : " f"{np.max(areas):.1f}" )

    print("==============================\n")

# ============================================================
# Draw image
# ============================================================

def visualize_image( image_path, annotations, category_lookup ):
    """
    Visualize a single image with its annotations.

    Parameters:
    - image_path: The file path to the image to be visualized.
    - annotations: A list of annotation dictionaries for the image, where each annotation contains a "bbox" (bounding box) and a "category_id".
    - category_lookup: A dictionary mapping category IDs to their corresponding category names, used to display the category name for each annotation.
        Example of category_lookup: {1: "person", 2: "bicycle", 3: "car", ...}.
    """

    image = cv2.imread(image_path)

    if image is None:
        print( f"Could not load:\n{image_path}" )
        return

    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )

    for ann in annotations:

        x, y, w, h = ann["bbox"]

        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        category_name = category_lookup[ ann["category_id"] ]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            category_name,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        os.path.basename(image_path)
    )
    plt.show()


# ============================================================
# Validation
# ============================================================

def validate_dataset( dataset_root, coco ):
    """
    Validate the COCO dataset by checking that all image paths exist.

    Parameters:
    - dataset_root: The root directory where the images are stored.
        The "file_name" field in the COCO "images" list is expected to be a relative path from this root.
    - coco: The COCO dataset loaded from the JSON file, containing an "images" list where each image has
        a "file_name" field that specifies the relative path to the image file from the dataset root.
    """

    print("Checking image paths...")

    missing = 0

    for image_info in coco["images"]:

        image_path = os.path.join( dataset_root, image_info["file_name"] )

        if not os.path.isfile(image_path):
            print(f"Missing:\n{image_path}")

            missing += 1

    print( f"Missing images: " f"{missing}" )

    if missing == 0:
        print("All image paths look valid.")


# ============================================================
# Main
# ============================================================

def main():

    # parse command-line arguments
    args = parse_arguments()
    dataset_root = os.path.abspath( args.dataset_root )
    coco_json = os.path.abspath( args.coco_json )

    # load coco dataset
    coco = load_coco(coco_json)

    print_statistics(coco)

    validate_dataset(dataset_root, coco)

    # build lookup tables for images and categories
    (image_lookup, category_lookup, annotations_per_image) = build_indices(coco)

    # randomly sample images and visualize
    image_ids = list( image_lookup.keys() )
    random.shuffle(image_ids)

    # limit to num_samples
    num_samples = min( args.num_samples, len(image_ids) )
    print( f"\nDisplaying " f"{num_samples} random images..." )

    # visualize samples
    for image_id in image_ids[:num_samples]:

        image_info = image_lookup[image_id]

        image_path = os.path.join(dataset_root,image_info["file_name"])

        annotations = ( annotations_per_image.get( image_id, [], ) )

        print(f"\nImage ID: {image_id}")

        print( f"Objects: " f"{len(annotations)}" )

        visualize_image( image_path, annotations, category_lookup )


if __name__ == "__main__":
    main()
