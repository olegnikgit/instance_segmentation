#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run RF-DETR instance segmentation on a folder of images.

The script:
    - loads a custom RF-DETR-Seg model
    - runs instance segmentation
    - saves individual binary masks
    - saves annotated overlay images
    - optionally displays each result
    - saves predictions to predictions.json

Example:

python rf_detr_instance_segmentation.py \
    --input_dir /path/to/input/images \
    --pretrain_weights /path/to/model/checkpoint.pth \
    --output_dir /path/to/output/results \
    --confidence 0.5 \
    --max_images 10 \
    --save_masks 1 \
    --save_overlay 1 \
    --display 1

@author: Olegs
"""

# ==========================================================
# IMPORTS
# ==========================================================

import argparse
import gc
import json
import os
import weakref

import cv2
import numpy as np
import torch
import supervision as sv

from rfdetr import RFDETRSegNano


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run RF-DETR instance segmentation."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--pretrain_weights",
        type=str,
        required=True,
        help="Path to the trained RF-DETR-Seg checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where results will be saved.",
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
        help=(
            "Maximum number of images to process. "
            "Default: process all images."
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="RF-DETR inference resolution.",
    )

    parser.add_argument(
        "--save_masks",
        type=int,
        default=1,
        help="Save individual binary masks: 1=yes, 0=no.",
    )

    parser.add_argument(
        "--save_overlay",
        type=int,
        default=1,
        help="Save segmentation overlay images: 1=yes, 0=no.",
    )

    parser.add_argument(
        "--display",
        type=int,
        default=1,
        help="Display results with OpenCV: 1=yes, 0=no.",
    )

    return parser.parse_args()


# ==========================================================
# GPU CLEANUP
# ==========================================================

def cleanup_gpu_memory(
    obj=None,
    verbose=False,
):
    """
    Release GPU memory when possible.
    """

    if not torch.cuda.is_available():

        if verbose:
            print(
                "[INFO] CUDA is not available."
            )

        return

    def memory_stats():
        allocated = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        return allocated, reserved

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    if verbose:

        allocated, reserved = memory_stats()

        print(
            f"[Before cleanup] "
            f"Allocated: {allocated:.2f} MB | "
            f"Reserved: {reserved:.2f} MB"
        )

    if obj is not None:

        ref = weakref.ref(obj)

        del obj

        if (
            ref() is not None
            and verbose
        ):
            print(
                "[WARNING] Object still exists."
            )

    gc.collect()

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    if verbose:

        allocated, reserved = memory_stats()

        print(
            f"[After cleanup] "
            f"Allocated: {allocated:.2f} MB | "
            f"Reserved: {reserved:.2f} MB"
        )


# ==========================================================
# GET CLASS NAME
# ==========================================================

def get_class_name(
    detections,
    object_id,
):
    """
    Get the class name associated with a detection.

    For custom RF-DETR models, the current RF-DETR API
    stores the resolved class names in:

        detections.data["class_name"]

    Falls back to class_<id> if unavailable.
    """

    if (
        hasattr(
            detections,
            "data",
        )
        and "class_name" in detections.data
    ):

        class_name = detections.data[
            "class_name"
        ][object_id]

        return str(class_name)

    if detections.class_id is not None:

        class_id = int(
            detections.class_id[
                object_id
            ]
        )

        return f"class_{class_id}"

    return "object"


# ==========================================================
# SAVE INDIVIDUAL MASKS
# ==========================================================

def save_instance_masks(
    detections,
    image,
    output_dir,
    base_name,
):
    """
    Save one binary PNG mask for every detected instance.

    Mask pixels:
        0   = background
        255 = object
    """

    if detections.mask is None:

        print(
            "  No segmentation masks returned."
        )

        return

    mask_dir = os.path.join(
        output_dir,
        "masks",
    )

    os.makedirs(
        mask_dir,
        exist_ok=True,
    )

    h, w = image.shape[:2]

    for object_id in range(
        len(detections)
    ):

        mask = detections.mask[
            object_id
        ]

        if mask is None:
            continue

        mask = np.asarray(mask)

        # --------------------------------------------------
        # Ensure 2D binary mask
        # --------------------------------------------------
        if mask.ndim != 2:

            raise ValueError(
                f"Unexpected mask shape: "
                f"{mask.shape}"
            )

        # --------------------------------------------------
        # Check mask dimensions
        # --------------------------------------------------
        if mask.shape != (
            h,
            w,
        ):

            # Normally RF-DETR returns masks at the
            # original image size. Resize if necessary.
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_uint8 = (
            mask > 0
        ).astype(
            np.uint8
        ) * 255

        class_name = (
            get_class_name(
                detections,
                object_id,
            )
        )

        score = float(
            detections.confidence[
                object_id
            ]
        )

        mask_filename = (
            f"{base_name}"
            f"_obj_{object_id:03d}"
            f"_{class_name}"
            f"_{score:.2f}.png"
        )

        mask_path = os.path.join(
            mask_dir,
            mask_filename,
        )

        cv2.imwrite(
            mask_path,
            mask_uint8,
        )


# ==========================================================
# COLLECT PREDICTIONS
# ==========================================================

def collect_predictions(
    detections,
    image_path,
    image_shape,
):
    """
    Convert RF-DETR detections into serializable
    Python dictionaries.

    Actual segmentation masks are not stored directly
    in JSON here. They are saved separately as PNG files.
    """

    predictions = []

    image_height, image_width = (
        image_shape[:2]
    )

    image_area = (
        image_width
        * image_height
    )

    for object_id in range(
        len(detections)
    ):

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------
        x1, y1, x2, y2 = (
            detections.xyxy[
                object_id
            ].tolist()
        )

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------
        if detections.confidence is not None:

            confidence = float(
                detections.confidence[
                    object_id
                ]
            )

        else:

            confidence = 0.0

        # --------------------------------------------------
        # Class
        # --------------------------------------------------
        if detections.class_id is not None:

            class_id = int(
                detections.class_id[
                    object_id
                ]
            )

        else:

            class_id = 0

        class_name = get_class_name(
            detections,
            object_id,
        )

        # --------------------------------------------------
        # Mask area
        # --------------------------------------------------
        mask_area = 0.0

        if (
            detections.mask is not None
            and detections.mask[
                object_id
            ] is not None
        ):

            mask = (
                detections.mask[
                    object_id
                ] > 0
            )

            mask_area = float(
                mask.sum()
            )

        # --------------------------------------------------
        # Prediction dictionary
        # --------------------------------------------------
        prediction = {

            "image_path": str(
                image_path
            ),

            "image_name": os.path.basename(
                image_path
            ),

            "image_width": int(
                image_width
            ),

            "image_height": int(
                image_height
            ),

            "class_id": class_id,

            "class_name": class_name,

            "object_id": int(
                object_id
            ),

            "bbox_xyxy": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],

            "bbox_xywh": [
                float(x1),
                float(y1),
                float(bbox_width),
                float(bbox_height),
            ],

            "confidence": confidence,

            "mask_area": mask_area,

            "mask_relative_area": (
                mask_area / image_area
                if image_area > 0
                else 0.0
            ),

            "bbox_area": (
                float(
                    bbox_width
                    * bbox_height
                )
            ),

            "aspect_ratio": (
                float(
                    bbox_width
                    / bbox_height
                )
                if bbox_height > 0
                else 0.0
            ),
        }

        predictions.append(
            prediction
        )

    return predictions


# ==========================================================
# CREATE OVERLAY
# ==========================================================

def create_overlay(
    image_rgb,
    detections,
):
    """
    Create a visualization containing:
        - instance masks
        - bounding boxes
        - class labels
        - confidence
    """

    annotated_image = image_rgb.copy()

    # ------------------------------------------------------
    # Mask annotation
    # ------------------------------------------------------
    mask_annotator = sv.MaskAnnotator()

    annotated_image = (
        mask_annotator.annotate(
            annotated_image,
            detections,
        )
    )

    # ------------------------------------------------------
    # Bounding-box annotation
    # ------------------------------------------------------
    box_annotator = sv.BoxAnnotator()

    annotated_image = (
        box_annotator.annotate(
            annotated_image,
            detections,
        )
    )

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------
    labels = []

    for object_id in range(
        len(detections)
    ):

        class_name = (
            get_class_name(
                detections,
                object_id,
            )
        )

        if detections.confidence is not None:

            confidence = float(
                detections.confidence[
                    object_id
                ]
            )

        else:

            confidence = 0.0

        labels.append(
            f"{class_name} "
            f"{confidence:.2f}"
        )

    label_annotator = (
        sv.LabelAnnotator()
    )

    annotated_image = (
        label_annotator.annotate(
            annotated_image,
            detections,
            labels,
        )
    )

    return annotated_image


# ==========================================================
# SAVE OVERLAY
# ==========================================================

def save_overlay(
    image_rgb,
    detections,
    output_dir,
    base_name,
):
    """
    Save annotated RGB image as PNG.
    """

    overlay_dir = os.path.join(
        output_dir,
        "overlay",
    )

    os.makedirs(
        overlay_dir,
        exist_ok=True,
    )

    annotated_image = create_overlay(
        image_rgb,
        detections,
    )

    output_path = os.path.join(
        overlay_dir,
        f"{base_name}.png",
    )

    # Convert RGB → BGR for OpenCV.
    annotated_bgr = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        output_path,
        annotated_bgr,
    )

    return annotated_image


# ==========================================================
# DISPLAY
# ==========================================================

def display_result(
    image_rgb,
    window_name,
):
    """
    Display an RGB image with OpenCV.

    Press:
        q / ESC → continue to next image
    """

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imshow(
        window_name,
        image_bgr,
    )

    while True:

        key = cv2.waitKey(50) & 0xFF

        if key in (
            ord("q"),
            27,
            ord(" "),
            ord("\n"),
        ):
            break


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # Check paths
    # ------------------------------------------------------
    if not os.path.isdir(
        args.input_dir
    ):
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{args.input_dir}"
        )

    if not os.path.isfile(
        args.pretrain_weights
    ):
        raise FileNotFoundError(
            f"Model weights do not exist:\n"
            f"{args.pretrain_weights}"
        )

    # ------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------
    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    if args.save_masks:

        os.makedirs(
            os.path.join(
                args.output_dir,
                "masks",
            ),
            exist_ok=True,
        )

    if args.save_overlay:

        os.makedirs(
            os.path.join(
                args.output_dir,
                "overlay",
            ),
            exist_ok=True,
        )

    # ------------------------------------------------------
    # Find images
    # ------------------------------------------------------
    valid_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    )

    image_paths = sorted(
        os.path.join(
            args.input_dir,
            filename,
        )
        for filename in os.listdir(
            args.input_dir
        )
        if filename.lower().endswith(
            valid_extensions
        )
    )

    if args.max_images is not None:

        image_paths = (
            image_paths[
                :args.max_images
            ]
        )

    print(
        f"Found {len(image_paths)} images."
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            "No images found."
        )

    # ------------------------------------------------------
    # GPU information
    # ------------------------------------------------------
    print("")
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ------------------------------------------------------
    # Cleanup before model loading
    # ------------------------------------------------------
    cleanup_gpu_memory(
        verbose=True
    )

    # ------------------------------------------------------
    # Load RF-DETR segmentation model
    # ------------------------------------------------------
    print("")
    print(
        "Loading RF-DETR Seg-Nano..."
    )

    model = RFDETRSegNano(
        resolution=args.resolution,
        pretrain_weights=args.pretrain_weights,
    )

    print(
        "Model loaded."
    )

    # ------------------------------------------------------
    # Optimize for inference
    # ------------------------------------------------------
    try:

        model.optimize_for_inference()

        print(
            "Model optimized for inference."
        )

    except AttributeError:

        print(
            "optimize_for_inference() "
            "is not available; continuing."
        )

    # ------------------------------------------------------
    # Process images
    # ------------------------------------------------------
    all_predictions = []

    try:

        for index, image_path in enumerate(
            image_paths
        ):

            filename = os.path.basename(
                image_path
            )

            base_name = os.path.splitext(
                filename
            )[0]

            print("")
            print(
                "=" * 70
            )
            print(
                f"[{index + 1}/{len(image_paths)}] "
                f"{filename}"
            )

            # --------------------------------------------------
            # Read image
            # --------------------------------------------------
            image_bgr = cv2.imread(
                image_path
            )

            if image_bgr is None:

                print(
                    f"[WARNING] Could not read "
                    f"{image_path}"
                )

                continue

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            # --------------------------------------------------
            # RF-DETR segmentation inference
            # --------------------------------------------------
            detections = model.predict(
                image_rgb,
                threshold=args.confidence,
            )

            print(
                f"Detected {len(detections)} "
                f"instances."
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

                confidence = float(
                    detections.confidence[
                        object_id
                    ]
                )

                print(
                    f"  Object {object_id}: "
                    f"{class_name} "
                    f"(confidence={confidence:.3f})"
                )

            # --------------------------------------------------
            # Collect predictions
            # --------------------------------------------------
            predictions = (
                collect_predictions(
                    detections=detections,
                    image_path=image_path,
                    image_shape=image_rgb.shape,
                )
            )

            all_predictions.extend(
                predictions
            )

            # --------------------------------------------------
            # Save actual masks
            # --------------------------------------------------
            if args.save_masks:

                save_instance_masks(
                    detections=detections,
                    image=image_rgb,
                    output_dir=args.output_dir,
                    base_name=base_name,
                )

            # --------------------------------------------------
            # Create/save overlay
            # --------------------------------------------------
            if args.save_overlay:

                annotated_image = (
                    save_overlay(
                        image_rgb=image_rgb,
                        detections=detections,
                        output_dir=args.output_dir,
                        base_name=base_name,
                    )
                )

            else:

                annotated_image = (
                    create_overlay(
                        image_rgb,
                        detections,
                    )
                )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------
            if args.display:

                display_result(
                    image_rgb=annotated_image,
                    window_name=(
                        "RF-DETR Instance "
                        "Segmentation"
                    ),
                )

    finally:

        if args.display:
            cv2.destroyAllWindows()

        cleanup_gpu_memory(
            model,
            verbose=True,
        )

    # ------------------------------------------------------
    # Save prediction information
    # ------------------------------------------------------
    predictions_path = os.path.join(
        args.output_dir,
        "predictions.json",
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_predictions,
            f,
            indent=2,
        )

    print("")
    print(
        "=" * 70
    )
    print(
        "Finished."
    )
    print(
        f"Images processed    : "
        f"{len(image_paths)}"
    )
    print(
        f"Instances detected  : "
        f"{len(all_predictions)}"
    )
    print(
        f"Predictions JSON    : "
        f"{predictions_path}"
    )

    if args.save_masks:

        print(
            f"Masks directory     : "
            f"{os.path.join(args.output_dir, 'masks')}"
        )

    if args.save_overlay:

        print(
            f"Overlay directory   : "
            f"{os.path.join(args.output_dir, 'overlay')}"
        )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()


