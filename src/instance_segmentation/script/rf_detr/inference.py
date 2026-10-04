#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run RF-DETR instance segmentation on a folder of images.

The output JSON follows the same annotation structure as the
provided dataset annotation format:

Top level:
    info
    categories
    videos
    images
    annotations

Prediction annotation:
    id
    image_id
    category_id
    instance_id
    bbox
    bbox_type
    confidence
    iscrowd
    ignore
    segmentation
    segmentation_type
    area

Inputs:
    --model_weights : path to trained RF-DETR model weights
    --context_dir   : directory containing COCO-style annotations
    --predict_dir   : directory containing images to predict
    --output_json   : output JSON path

Optional:
    --save_masks
    --save_overlay
    --max_images

Example:

python inference.py \
    --model_weights /path/to/checkpoint_best_total.pth \
    --context_dir /path/to/context \
    --predict_dir /path/to/images \
    --output_json /path/to/predictions/predictions.json \
    --confidence 0.5 \
    --save_masks \
    --save_overlay
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from rfdetr import RFDETR


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR instance segmentation."
    )

    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to trained RF-DETR model checkpoint.",
    )

    parser.add_argument(
        "--context_dir",
        type=str,
        required=True,
        help="Directory containing the reference COCO annotation JSON.",
    )

    parser.add_argument(
        "--predict_dir",
        type=str,
        required=True,
        help="Directory containing images to predict.",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output prediction JSON.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process.",
    )

    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Save individual binary instance masks.",
    )

    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save segmentation overlay images.",
    )

    return parser.parse_args()


# ==========================================================
# JSON HELPERS
# ==========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_annotation_file(directory):
    directory = Path(directory)

    for filename in (
        "_annotations.coco.json",
        "annotations.json",
        "annotations.coco.json",
    ):
        path = directory / filename

        if path.is_file():
            return path

    return None


# ==========================================================
# MASK HELPERS
# ==========================================================

def mask_to_rle(mask):
    """
    Convert a binary mask to compressed COCO RLE.

    Output:
        {
            "size": [height, width],
            "counts": "..."
        }
    """

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    rle = mask_utils.encode(
        np.asfortranarray(mask)
    )

    counts = rle["counts"]

    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    return {
        "size": list(rle["size"]),
        "counts": counts,
    }


# ==========================================================
# RF-DETR CLASS HELPERS
# ==========================================================

def get_class_name(detections, object_id):
    if (
        hasattr(detections, "data")
        and "class_name" in detections.data
    ):
        return str(
            detections.data["class_name"][object_id]
        )

    if detections.class_id is not None:
        return (
            f"class_"
            f"{int(detections.class_id[object_id])}"
        )

    return "object"


def get_category_map(categories):
    return {
        str(category["name"]): int(category["id"])
        for category in categories
    }


def get_category_id(
    detections,
    object_id,
    category_map,
):
    class_name = get_class_name(
        detections,
        object_id,
    )

    if class_name in category_map:
        return category_map[class_name]

    if detections.class_id is not None:
        class_id = int(
            detections.class_id[object_id]
        )

        if class_id in category_map.values():
            return class_id

    raise ValueError(
        f"Could not map RF-DETR class "
        f"'{class_name}' to a dataset category."
    )


# ==========================================================
# IMAGE RECORDS
# ==========================================================

def create_image_records(
    image_paths,
    prediction_data,
    context_data,
):
    """
    Preserve existing image records whenever possible.

    Priority:
        1. annotation JSON in predict_dir
        2. matching image record in context annotation
        3. create a new record using the reference schema
    """

    prediction_images = {}

    if prediction_data is not None:
        prediction_images = {
            image["file_name"]: image
            for image in prediction_data.get(
                "images",
                [],
            )
        }

    context_images = {
        image["file_name"]: image
        for image in context_data.get(
            "images",
            [],
        )
    }

    image_records = []
    image_ids = {}

    next_image_id = 1

    all_existing_ids = [
        int(image["id"])
        for image in context_data.get(
            "images",
            [],
        )
        if "id" in image
    ]

    all_existing_ids.extend(
        int(image["id"])
        for image in prediction_data.get(
            "images",
            [],
        )
        if "id" in image
    ) if prediction_data is not None else None

    if all_existing_ids:
        next_image_id = max(
            all_existing_ids
        ) + 1

    for image_path in image_paths:

        filename = image_path.name

        if filename in prediction_images:

            source = prediction_images[
                filename
            ]

            image_record = copy.deepcopy(
                source
            )

            image_id = int(
                source["id"]
            )

        elif filename in context_images:

            source = context_images[
                filename
            ]

            image_record = copy.deepcopy(
                source
            )

            image_id = int(
                source["id"]
            )

        else:

            with Image.open(
                image_path
            ) as image:
                width, height = image.size

            # --------------------------------------------------
            # Create record with exactly the same field structure
            # as the provided annotation format.
            # --------------------------------------------------

            image_id = next_image_id
            next_image_id += 1

            image_record = {
                "id": image_id,
                "video_id": 0,
                "file_name": filename,
                "width": int(width),
                "height": int(height),
                "reviewed_categories": [],
                "tags": [],
            }

        image_ids[filename] = image_id
        image_records.append(
            image_record
        )

    return (
        image_records,
        image_ids,
    )


# ==========================================================
# SAVE INDIVIDUAL MASKS
# ==========================================================

def save_instance_masks(
    detections,
    output_dir,
    image_stem,
):
    """
    Save one binary PNG per detected instance.

    0   = background
    255 = instance
    """

    if detections.mask is None:
        return

    mask_dir = (
        Path(output_dir)
        / "masks"
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for object_id in range(
        len(detections)
    ):

        mask = detections.mask[
            object_id
        ]

        if mask is None:
            continue

        mask = (
            np.asarray(mask) > 0
        ).astype(
            np.uint8
        ) * 255

        Image.fromarray(
            mask,
            mode="L",
        ).save(
            mask_dir
            / (
                f"{image_stem}"
                f"_obj_{object_id:03d}.png"
            )
        )


# ==========================================================
# SAVE OVERLAY
# ==========================================================

def save_overlay(
    image_rgb,
    detections,
    output_dir,
    image_stem,
):
    """
    Save image with:
        - instance masks
        - bounding boxes
        - class names
        - confidence scores

    Uses PIL only.
    """

    if detections.mask is None:
        return

    overlay_dir = (
        Path(output_dir)
        / "overlay"
    )

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = Image.fromarray(
        image_rgb
    ).convert("RGBA")

    mask_layer = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(
        mask_layer
    )

    colors = [
        (255, 80, 80, 110),
        (80, 160, 255, 110),
        (80, 220, 120, 110),
        (255, 190, 60, 110),
        (190, 100, 255, 110),
        (60, 220, 220, 110),
    ]

    for object_id in range(
        len(detections)
    ):

        mask = detections.mask[
            object_id
        ]

        if mask is None:
            continue

        mask = np.asarray(
            mask
        ).astype(bool)

        color = colors[
            object_id
            % len(colors)
        ]

        solid_color = color[:3]

        mask_rgba = np.zeros(
            (
                mask.shape[0],
                mask.shape[1],
                4,
            ),
            dtype=np.uint8,
        )

        mask_rgba[mask] = color

        mask_image = Image.fromarray(
            mask_rgba,
            mode="RGBA",
        )

        mask_layer.alpha_composite(
            mask_image
        )

        x1, y1, x2, y2 = map(
            float,
            detections.xyxy[
                object_id
            ],
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=solid_color + (255,),
            width=3,
        )

        class_name = get_class_name(
            detections,
            object_id,
        )

        score = (
            float(
                detections.confidence[
                    object_id
                ]
            )
            if detections.confidence is not None
            else 0.0
        )

        label = (
            f"{class_name} "
            f"{score:.2f}"
        )

        draw.text(
            (
                x1 + 4,
                max(
                    0,
                    y1 - 18,
                ),
            ),
            label,
            fill=solid_color + (255,),
        )

    result = Image.alpha_composite(
        base,
        mask_layer,
    ).convert("RGB")

    result.save(
        overlay_dir
        / f"{image_stem}.png"
    )


# ==========================================================
# COLLECT PREDICTIONS
# ==========================================================

def collect_predictions(
    detections,
    image_id,
    prediction_id_start,
    category_map,
):
    """
    Convert RF-DETR detections to the exact annotation
    field structure used by the supplied dataset.

    Prediction fields:

        id
        image_id
        category_id
        instance_id
        bbox
        bbox_type
        confidence
        iscrowd
        ignore
        segmentation
        segmentation_type
        area
    """

    predictions = []

    next_id = prediction_id_start

    for object_id in range(
        len(detections)
    ):

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------

        x1, y1, x2, y2 = map(
            float,
            detections.xyxy[
                object_id
            ],
        )

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        bbox = [
            x1,
            y1,
            bbox_width,
            bbox_height,
        ]

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence = (
            float(
                detections.confidence[
                    object_id
                ]
            )
            if detections.confidence is not None
            else 0.0
        )

        # --------------------------------------------------
        # Category
        # --------------------------------------------------

        category_id = get_category_id(
            detections,
            object_id,
            category_map,
        )

        # --------------------------------------------------
        # Segmentation
        # --------------------------------------------------

        if (
            detections.mask is None
            or detections.mask[
                object_id
            ] is None
        ):
            raise RuntimeError(
                f"RF-DETR did not return a "
                f"segmentation mask for object "
                f"{object_id}."
            )

        mask = np.asarray(
            detections.mask[
                object_id
            ]
        ).astype(bool)

        segmentation = mask_to_rle(
            mask
        )

        area = float(
            mask.sum()
        )

        # --------------------------------------------------
        # Dataset-compatible annotation
        # --------------------------------------------------

        prediction = {
            "id": int(next_id),
            "image_id": int(image_id),
            "category_id": int(category_id),
            "instance_id": int(next_id),
            "bbox": bbox,
            "bbox_type": "xywh",
            "confidence": confidence,
            "iscrowd": 0,
            "ignore": 0,
            "segmentation": segmentation,
            "segmentation_type": "rle",
            "area": area,
        }

        predictions.append(
            prediction
        )

        next_id += 1

    return (
        predictions,
        next_id,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    args = parse_args()

    model_weights = Path(
        args.model_weights
    ).expanduser().resolve()

    context_dir = Path(
        args.context_dir
    ).expanduser().resolve()

    predict_dir = Path(
        args.predict_dir
    ).expanduser().resolve()

    output_json = Path(
        args.output_json
    ).expanduser().resolve()

    # ------------------------------------------------------
    # Checks
    # ------------------------------------------------------

    if not model_weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found:\n"
            f"{model_weights}"
        )

    if not context_dir.is_dir():
        raise FileNotFoundError(
            f"Context directory not found:\n"
            f"{context_dir}"
        )

    if not predict_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found:\n"
            f"{predict_dir}"
        )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Context annotation file
    # ------------------------------------------------------

    context_ann_path = find_annotation_file(
        context_dir
    )

    if context_ann_path is None:
        raise FileNotFoundError(
            "No annotation JSON found in context directory:\n"
            f"{context_dir}"
        )

    context_data = load_json(
        context_ann_path
    )

    categories = copy.deepcopy(
        context_data.get(
            "categories",
            [],
        )
    )

    if not categories:
        raise RuntimeError(
            "No categories found in context annotation file."
        )

    category_map = get_category_map(
        categories
    )

    # ------------------------------------------------------
    # Optional annotation file in prediction directory
    # ------------------------------------------------------

    predict_ann_path = find_annotation_file(
        predict_dir
    )

    prediction_data = None

    if predict_ann_path is not None:
        prediction_data = load_json(
            predict_ann_path
        )

    # ------------------------------------------------------
    # Find images
    # ------------------------------------------------------

    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = sorted(
        path
        for path in predict_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in valid_extensions
        )
    )

    if args.max_images is not None:
        image_paths = image_paths[
            :args.max_images
        ]

    if not image_paths:
        raise RuntimeError(
            f"No images found in:\n"
            f"{predict_dir}"
        )

    # ------------------------------------------------------
    # Build exact-format image records
    # ------------------------------------------------------

    image_records, image_ids = (
        create_image_records(
            image_paths=image_paths,
            prediction_data=prediction_data,
            context_data=context_data,
        )
    )

    # ------------------------------------------------------
    # Display environment
    # ------------------------------------------------------

    print(
        f"Found {len(image_paths)} images."
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print(
        "\nLoading RF-DETR model..."
    )

    model = RFDETR.from_checkpoint(
        str(model_weights)
    )

    print(
        "Model loaded."
    )

    if hasattr(
        model,
        "optimize_for_inference",
    ):
        try:
            model.optimize_for_inference()

            print(
                "Model optimized for inference."
            )

        except Exception as exc:
            print(
                "Warning: could not optimize "
                f"model: {exc}"
            )

    # ------------------------------------------------------
    # Run inference
    # ------------------------------------------------------

    all_predictions = []

    next_prediction_id = 1

    try:

        for index, image_path in enumerate(
            image_paths,
            1,
        ):

            image_id = image_ids[
                image_path.name
            ]

            image_stem = (
                image_path.stem
            )

            print(
                f"\n[{index}/{len(image_paths)}] "
                f"{image_path.name}"
            )

            image_rgb = np.asarray(
                Image.open(
                    image_path
                ).convert("RGB")
            )

            detections = model.predict(
                str(image_path),
                threshold=args.confidence,
            )

            print(
                f"  Detected "
                f"{len(detections)} instances."
            )

            # --------------------------------------------------
            # Print detections
            # --------------------------------------------------

            for object_id in range(
                len(detections)
            ):

                class_name = (
                    get_class_name(
                        detections,
                        object_id,
                    )
                )

                confidence = (
                    float(
                        detections.confidence[
                            object_id
                        ]
                    )
                    if detections.confidence is not None
                    else 0.0
                )

                print(
                    f"    Object {object_id}: "
                    f"{class_name} "
                    f"(confidence="
                    f"{confidence:.3f})"
                )

            # --------------------------------------------------
            # Convert predictions
            # --------------------------------------------------

            predictions, next_prediction_id = (
                collect_predictions(
                    detections=detections,
                    image_id=image_id,
                    prediction_id_start=(
                        next_prediction_id
                    ),
                    category_map=category_map,
                )
            )

            all_predictions.extend(
                predictions
            )

            # --------------------------------------------------
            # Save individual masks
            # --------------------------------------------------

            if args.save_masks:
                save_instance_masks(
                    detections=detections,
                    output_dir=output_json.parent,
                    image_stem=image_stem,
                )

            # --------------------------------------------------
            # Save overlay
            # --------------------------------------------------

            if args.save_overlay:
                save_overlay(
                    image_rgb=image_rgb,
                    detections=detections,
                    output_dir=output_json.parent,
                    image_stem=image_stem,
                )

    finally:

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ======================================================
    # BUILD FINAL JSON
    # ======================================================

    # ------------------------------------------------------
    # Preserve dataset-level metadata exactly
    # ------------------------------------------------------

    result = {
        "info": copy.deepcopy(
            context_data.get(
                "info",
                {},
            )
        ),
        "categories": categories,
        "videos": copy.deepcopy(
            context_data.get(
                "videos",
                [],
            )
        ),
        "images": image_records,
        "annotations": all_predictions,
    }

    # ------------------------------------------------------
    # If prediction folder had its own annotation JSON,
    # preserve its videos/info when available.
    # This is especially useful for cross-validation.
    # ------------------------------------------------------

    if prediction_data is not None:

        if "info" in prediction_data:
            result["info"] = copy.deepcopy(
                prediction_data["info"]
            )

        if "videos" in prediction_data:
            result["videos"] = copy.deepcopy(
                prediction_data["videos"]
            )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Finished."
    )

    print(
        f"Images processed   : "
        f"{len(image_paths)}"
    )

    print(
        f"Instances detected : "
        f"{len(all_predictions)}"
    )

    print(
        f"Predictions JSON   : "
        f"{output_json}"
    )

    if args.save_masks:

        print(
            f"Masks directory    : "
            f"{output_json.parent / 'masks'}"
        )

    if args.save_overlay:

        print(
            f"Overlay directory  : "
            f"{output_json.parent / 'overlay'}"
        )


if __name__ == "__main__":
    main()
