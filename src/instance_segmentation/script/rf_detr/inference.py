#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run RF-DETR instance segmentation on a folder of images.

Outputs:
    - COCO-compatible predictions.json
    - optional individual binary instance masks
    - optional mask/bbox/label overlay images

No OpenCV, supervision, or gc are required.

Example:
python rf_detr_instance_segmentation_inference.py \
    --model_weights /path/to/checkpoint.pth \
    --context_dir /path/to/train \
    --predict_dir /path/to/valid \
    --output_json /path/to/results/predictions.json \
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
    parser = argparse.ArgumentParser(description="Run RF-DETR instance segmentation.")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to trained RF-DETR-Seg checkpoint.")
    parser.add_argument("--context_dir", type=str, required=True,
                        help="Directory containing training COCO annotation file/categories.")
    parser.add_argument("--predict_dir", type=str, required=True,
                        help="Directory containing images to predict.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output COCO predictions JSON.")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process.")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save individual binary instance masks.")
    parser.add_argument("--save_overlay", action="store_true",
                        help="Save images with segmentation masks overlaid.")
    return parser.parse_args()


# ==========================================================
# HELPERS
# ==========================================================

def find_annotation_file(directory):
    directory = Path(directory)
    for name in ("_annotations.coco.json", "annotations.json", "annotations.coco.json"):
        path = directory / name
        if path.is_file():
            return path
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mask_to_rle(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask))
    counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return {"size": list(rle["size"]), "counts": counts}


def get_class_name(detections, object_id):
    if hasattr(detections, "data") and "class_name" in detections.data:
        return str(detections.data["class_name"][object_id])
    if detections.class_id is not None:
        return f"class_{int(detections.class_id[object_id])}"
    return "object"


def get_category_map(categories):
    return {str(c["name"]): int(c["id"]) for c in categories}


def get_class_category_id(detections, object_id, category_map):
    class_name = get_class_name(detections, object_id)
    if class_name in category_map:
        return category_map[class_name]

    if detections.class_id is not None:
        return int(detections.class_id[object_id])

    return 0


# ==========================================================
# SAVE INDIVIDUAL MASKS
# ==========================================================

def save_instance_masks(detections, output_dir, image_stem):
    if detections.mask is None:
        return

    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for object_id in range(len(detections)):
        mask = detections.mask[object_id]
        if mask is None:
            continue

        mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(
            mask_dir / f"{image_stem}_obj_{object_id:03d}.png"
        )


# ==========================================================
# SAVE OVERLAY
# ==========================================================

def save_overlay(image_rgb, detections, output_dir, image_stem):
    """
    Save RGB image with semi-transparent instance masks,
    bounding boxes, class names, and confidence scores.
    """

    if detections.mask is None:
        return

    overlay_dir = Path(output_dir) / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    base = Image.fromarray(image_rgb).convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    colors = [
        (255, 80, 80, 110),
        (80, 160, 255, 110),
        (80, 220, 120, 110),
        (255, 190, 60, 110),
        (190, 100, 255, 110),
        (60, 220, 220, 110),
    ]

    for object_id in range(len(detections)):
        mask = detections.mask[object_id]
        if mask is None:
            continue

        mask = np.asarray(mask).astype(bool)
        color = colors[object_id % len(colors)]
        solid_color = color[:3]

        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask] = color
        mask_image = Image.fromarray(mask_rgba, mode="RGBA")
        mask_layer.alpha_composite(mask_image)

        x1, y1, x2, y2 = map(float, detections.xyxy[object_id])
        draw.rectangle((x1, y1, x2, y2), outline=solid_color + (255,), width=3)

        class_name = get_class_name(detections, object_id)
        score = float(detections.confidence[object_id]) if detections.confidence is not None else 0.0
        label = f"{class_name} {score:.2f}"

        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=solid_color + (255,))

    result = Image.alpha_composite(base, mask_layer).convert("RGB")
    result.save(overlay_dir / f"{image_stem}.png")


# ==========================================================
# COLLECT COCO PREDICTIONS
# ==========================================================

def collect_predictions(detections, image_id, image_path, category_map):
    predictions = []

    image = Image.open(image_path)
    image_width, image_height = image.size

    for object_id in range(len(detections)):
        x1, y1, x2, y2 = map(float, detections.xyxy[object_id])
        bbox_w, bbox_h = x2 - x1, y2 - y1

        confidence = (
            float(detections.confidence[object_id])
            if detections.confidence is not None else 0.0
        )

        category_id = get_class_category_id(detections, object_id, category_map)

        prediction = {
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [x1, y1, bbox_w, bbox_h],
            "score": confidence,
            "confidence": confidence,
        }

        if detections.mask is not None and detections.mask[object_id] is not None:
            mask = np.asarray(detections.mask[object_id]).astype(bool)
            prediction["segmentation"] = mask_to_rle(mask)
            prediction["area"] = float(mask.sum())
        else:
            prediction["segmentation"] = None
            prediction["area"] = float(bbox_w * bbox_h)

        predictions.append(prediction)

    return predictions


# ==========================================================
# MAIN
# ==========================================================

def main():
    args = parse_args()

    model_weights = Path(args.model_weights)
    context_dir = Path(args.context_dir)
    predict_dir = Path(args.predict_dir)
    output_json = Path(args.output_json)

    if not model_weights.is_file():
        raise FileNotFoundError(f"Model weights not found:\n{model_weights}")
    if not context_dir.is_dir():
        raise FileNotFoundError(f"Context directory not found:\n{context_dir}")
    if not predict_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found:\n{predict_dir}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    context_ann_path = find_annotation_file(context_dir)
    if context_ann_path is None:
        raise FileNotFoundError(f"No COCO annotation file found in:\n{context_dir}")

    context_data = load_json(context_ann_path)
    categories = copy.deepcopy(context_data.get("categories", []))
    category_map = get_category_map(categories)

    predict_ann_path = find_annotation_file(predict_dir)
    predict_data = load_json(predict_ann_path) if predict_ann_path else None

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(p for p in predict_dir.iterdir() if p.suffix.lower() in valid_extensions)

    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        raise RuntimeError(f"No images found in:\n{predict_dir}")

    # Preserve the original COCO image IDs whenever ground truth is available.
    image_records = {}
    if predict_data:
        image_records = {img["file_name"]: img for img in predict_data.get("images", [])}

    image_ids = {}
    next_image_id = 1

    for image_path in image_paths:
        record = image_records.get(image_path.name)

        if record is not None:
            image_ids[image_path.name] = int(record["id"])
        else:
            image_ids[image_path.name] = next_image_id
            next_image_id += 1

    print(f"Found {len(image_paths)} images.")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading RF-DETR model...")
    model = RFDETR.from_checkpoint(str(model_weights))
    print("Model loaded.")

    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
            print("Model optimized for inference.")
        except Exception as exc:
            print(f"Warning: could not optimize model: {exc}")

    all_predictions = []

    try:
        for index, image_path in enumerate(image_paths, 1):
            image_id = image_ids[image_path.name]
            image_stem = image_path.stem

            print(f"\n[{index}/{len(image_paths)}] {image_path.name}")

            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            detections = model.predict(str(image_path), threshold=args.confidence)

            print(f"  Detected {len(detections)} instances.")

            for object_id in range(len(detections)):
                class_name = get_class_name(detections, object_id)
                score = float(detections.confidence[object_id])
                print(f"    Object {object_id}: {class_name} (confidence={score:.3f})")

            all_predictions.extend(
                collect_predictions(detections, image_id, image_path, category_map)
            )

            if args.save_masks:
                save_instance_masks(detections, output_json.parent, image_stem)

            if args.save_overlay:
                save_overlay(image_rgb, detections, output_json.parent, image_stem)

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "images": [
            image_records[name]
            if name in image_records
            else {
                "id": image_ids[name],
                "file_name": name,
                "width": int(Image.open(predict_dir / name).width),
                "height": int(Image.open(predict_dir / name).height),
            }
            for name in [p.name for p in image_paths]
        ],
        "annotations": all_predictions,
        "categories": categories,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("Finished.")
    print(f"Images processed   : {len(image_paths)}")
    print(f"Instances detected : {len(all_predictions)}")
    print(f"Predictions JSON   : {output_json}")

    if args.save_masks:
        print(f"Masks directory    : {output_json.parent / 'masks'}")

    if args.save_overlay:
        print(f"Overlay directory  : {output_json.parent / 'overlay'}")


if __name__ == "__main__":
    main()
