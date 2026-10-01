#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize COCO annotations including bounding boxes and instance masks.

Example:
    python visualize_coco.py \
        --annotations /path/to/_annotations.coco.json \
        --images_dir /path/to/images \
        --output_dir /path/to/visualized \
        --max_images 10
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
    )

    return parser.parse_args()


def decode_segmentation(segmentation, image_height, image_width):
    """
    Decode COCO segmentation into a binary mask.

    Supports:
        - Polygon segmentation
        - Uncompressed RLE
        - Compressed RLE

    Returns:
        mask: H x W uint8 mask
    """

    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # Polygon segmentation
    # ---------------------------------------------------------

    if isinstance(segmentation, list):

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            points = np.array(
                polygon,
                dtype=np.float32,
            ).reshape(-1, 2)

            points = np.round(points).astype(
                np.int32
            )

            cv2.fillPoly(
                mask,
                [points],
                1,
            )

        return mask

    # ---------------------------------------------------------
    # RLE segmentation
    # ---------------------------------------------------------

    if isinstance(segmentation, dict):

        # pycocotools is required for RLE
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise RuntimeError(
                "RLE segmentation detected, but pycocotools "
                "is not installed.\n\n"
                "Install it with:\n"
                "pip install pycocotools"
            )

        rle = segmentation.copy()

        # Compressed RLE uses a string.
        # pycocotools.decode() can handle it directly.
        decoded = mask_utils.decode(rle)

        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]

        return decoded.astype(np.uint8)

    return mask


def overlay_mask(
    image,
    mask,
    color,
    alpha=0.35,
):
    """Overlay a binary mask onto an image."""

    if mask is None:
        return

    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        return

    overlay = image.copy()

    overlay[mask_bool] = color

    image[:] = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
    )


def draw_mask_contour(
    image,
    mask,
    color,
):
    """Draw mask contour."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        2,
    )


def main():

    args = parse_args()

    annotations_path = Path(
        args.annotations
    )

    images_dir = Path(
        args.images_dir
    )

    # ---------------------------------------------------------
    # Load COCO JSON
    # ---------------------------------------------------------

    print(
        f"Loading annotations: {annotations_path}"
    )

    with open(
        annotations_path,
        "r",
        encoding="utf-8",
    ) as f:
        coco = json.load(f)

    images = coco.get(
        "images",
        []
    )

    annotations = coco.get(
        "annotations",
        []
    )

    categories = coco.get(
        "categories",
        []
    )

    print()
    print("COCO dataset:")
    print(f"  Images:      {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories:  {len(categories)}")
    print()

    # ---------------------------------------------------------
    # Check segmentation availability
    # ---------------------------------------------------------

    polygon_count = 0
    rle_count = 0
    no_mask_count = 0

    for annotation in annotations:

        segmentation = annotation.get(
            "segmentation"
        )

        if isinstance(segmentation, list):
            polygon_count += 1

        elif isinstance(segmentation, dict):
            rle_count += 1

        else:
            no_mask_count += 1

    print("Segmentation information:")
    print(f"  Polygon masks: {polygon_count}")
    print(f"  RLE masks:     {rle_count}")
    print(f"  No masks:      {no_mask_count}")
    print()

    if polygon_count == 0 and rle_count == 0:

        print(
            "WARNING: No segmentation masks were found "
            "in the COCO annotations."
        )

        print(
            "The JSON appears to contain bounding boxes only."
        )

    # ---------------------------------------------------------
    # Category mapping
    # ---------------------------------------------------------

    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    # ---------------------------------------------------------
    # Group annotations by image
    # ---------------------------------------------------------

    annotations_by_image = {}

    for annotation in annotations:

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = None

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Select images
    # ---------------------------------------------------------

    images_to_process = images

    if args.max_images is not None:

        images_to_process = images[
            :args.max_images
        ]

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    for index, image_info in enumerate(
        images_to_process
    ):

        image_id = image_info["id"]

        file_name = image_info[
            "file_name"
        ]

        image_path = images_dir / file_name

        if not image_path.exists():

            print(
                f"WARNING: Image not found: "
                f"{image_path}"
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read: "
                f"{image_path}"
            )

            continue

        height, width = image.shape[:2]

        image_annotations = (
            annotations_by_image.get(
                image_id,
                []
            )
        )

        # -----------------------------------------------------
        # Draw every instance
        # -----------------------------------------------------

        for annotation in image_annotations:

            category_id = annotation[
                "category_id"
            ]

            category_name = category_map.get(
                category_id,
                f"class_{category_id}",
            )

            # -------------------------------------------------
            # Instance segmentation mask
            # -------------------------------------------------

            segmentation = annotation.get(
                "segmentation"
            )

            if segmentation:

                mask = decode_segmentation(
                    segmentation,
                    height,
                    width,
                )

                # Use a different color for each instance
                # based on annotation ID.
                rng = np.random.default_rng(
                    annotation["id"]
                )

                color = tuple(
                    int(x)
                    for x in rng.integers(
                        50,
                        255,
                        size=3,
                    )
                )

                # OpenCV uses BGR
                overlay_mask(
                    image,
                    mask,
                    color,
                    alpha=0.35,
                )

                draw_mask_contour(
                    image,
                    mask,
                    color,
                )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            bbox = annotation.get(
                "bbox"
            )

            if bbox:

                x, y, w, h = bbox

                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                label = (
                    f"{category_name} "
                    f"({category_id})"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        print(
            f"[{index + 1}/{len(images_to_process)}] "
            f"{file_name}: "
            f"{len(image_annotations)} instances"
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        if output_dir:

            output_path = (
                output_dir / file_name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(output_path),
                image,
            )

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        display_image = image.copy()

        max_width = 1600
        max_height = 1000

        scale = min(
            max_width / width,
            max_height / height,
            1.0,
        )

        if scale < 1.0:

            display_image = cv2.resize(
                display_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

    #     cv2.imshow(
    #         "COCO Instance Segmentation",
    #         display_image,
    #     )

    #     key = cv2.waitKey(0) & 0xFF

    #     if key in (
    #         27,
    #         ord("q"),
    #     ):
    #         break

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
