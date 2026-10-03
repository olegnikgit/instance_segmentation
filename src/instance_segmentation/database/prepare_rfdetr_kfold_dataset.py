#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a K-fold COCO-style instance-segmentation dataset for RF-DETR.

For 10 images and 5 folds:
    fold_01: 8 train, 2 valid
    fold_02: 8 train, 2 valid
    ...
    fold_05: 8 train, 2 valid

Output:
output_dir/
├── fold_01/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── images
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── images
│   └── split_manifest.json
├── ...
├── fold_05/
└── cross_validation_manifest.json

Example:
python prepare_rfdetr_kfold_dataset.py \
    --annotations /path/to/annotations.json \
    --images_dir /path/to/images \
    --output_dir /path/to/rf_detr_5fold \
    --num_folds 5 \
    --seed 42

@author: Olegs
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a K-fold COCO instance-segmentation dataset for RF-DETR."
    )
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations.json.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_rle_validation", action="store_true", help="Skip segmentation-mask validation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing to a non-empty output directory.")
    return parser.parse_args()


# ==========================================================
# JSON / DATA VALIDATION
# ==========================================================

def load_json(path: Path):
    print(f"Loading annotations: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Input JSON does not contain required key: '{key}'")
    return data


def validate_categories(categories):
    if not categories:
        raise ValueError("No categories found in annotations.")

    category_ids = set()
    print("\nCategories:")

    for category in categories:
        category_id, category_name = category["id"], category["name"]
        if category_id in category_ids:
            raise ValueError(f"Duplicate category ID: {category_id}")
        category_ids.add(category_id)
        print(f"  id={category_id}: {category_name}")

    return category_ids


def build_image_index(images):
    images_by_id = {image["id"]: image for image in images}
    if len(images_by_id) != len(images):
        raise ValueError("Duplicate image IDs detected.")
    return images_by_id


def resolve_image_path(images_dir: Path, file_name: str):
    path = images_dir / file_name
    if path.exists():
        return path

    path = images_dir / Path(file_name).name
    return path if path.exists() else None


def validate_images(images, images_dir: Path):
    print("\nValidating images...")
    resolved_paths = {}

    for image in images:
        image_id = image["id"]
        image_path = resolve_image_path(images_dir, image["file_name"])

        if image_path is None:
            raise FileNotFoundError(
                f"Image not found:\n"
                f"  image_id : {image_id}\n"
                f"  file_name: {image['file_name']}\n"
                f"  searched : {images_dir}"
            )

        try:
            with Image.open(image_path) as pil_image:
                actual_width, actual_height = pil_image.size
        except Exception as exc:
            raise RuntimeError(f"Could not open image '{image_path}': {exc}") from exc

        if (actual_width, actual_height) != (image["width"], image["height"]):
            raise ValueError(
                f"Image dimension mismatch for '{image['file_name']}':\n"
                f"  JSON : {image['width']} x {image['height']}\n"
                f"  file : {actual_width} x {actual_height}"
            )

        resolved_paths[image_id] = image_path

    print(f"  Validated {len(images)} images successfully.")
    return resolved_paths


# ==========================================================
# SEGMENTATION VALIDATION
# ==========================================================

def validate_rle_annotation(annotation, image_width, image_height):
    segmentation = annotation.get("segmentation")

    if segmentation is None:
        raise ValueError(f"Annotation {annotation['id']} has no segmentation.")

    # Polygon segmentation
    if isinstance(segmentation, list):
        if mask_utils is None:
            return None
        rles = mask_utils.frPyObjects(segmentation, image_height, image_width)
        return int(mask_utils.decode(rles).sum())

    # RLE segmentation
    if not isinstance(segmentation, dict):
        raise ValueError(
            f"Annotation {annotation['id']} has unsupported segmentation type: "
            f"{type(segmentation)}"
        )

    if "size" not in segmentation or "counts" not in segmentation:
        raise ValueError(f"Invalid RLE segmentation in annotation {annotation['id']}")

    rle_height, rle_width = segmentation["size"]

    if (rle_height, rle_width) != (image_height, image_width):
        raise ValueError(
            f"RLE size mismatch in annotation {annotation['id']}:\n"
            f"  RLE   : {rle_width} x {rle_height}\n"
            f"  image : {image_width} x {image_height}"
        )

    if mask_utils is None:
        return None

    try:
        decoded = mask_utils.decode(segmentation)
    except Exception as exc:
        raise ValueError(f"Could not decode RLE for annotation {annotation['id']}: {exc}") from exc

    decoded_area = int(decoded.sum())

    if decoded_area <= 0:
        raise ValueError(f"Annotation {annotation['id']} has an empty mask.")

    return decoded_area


def validate_annotations(annotations, images_by_id, category_ids, disable_rle_validation=False):
    print("\nValidating annotations...")

    annotation_ids = set()
    stats = {"num_annotations": 0, "num_rle": 0, "num_polygon": 0}

    for annotation in annotations:
        annotation_id = annotation["id"]

        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {annotation_id}")
        annotation_ids.add(annotation_id)

        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in images_by_id:
            raise ValueError(f"Annotation {annotation_id} references unknown image_id={image_id}")

        if category_id not in category_ids:
            raise ValueError(f"Annotation {annotation_id} references unknown category_id={category_id}")

        image = images_by_id[image_id]
        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Invalid bbox in annotation {annotation_id}: {bbox}")

        _, _, width, height = bbox

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid bbox dimensions in annotation {annotation_id}: {bbox}")

        segmentation = annotation.get("segmentation")

        if segmentation is None:
            raise ValueError(f"Annotation {annotation_id} has no segmentation.")

        if isinstance(segmentation, dict):
            stats["num_rle"] += 1
        elif isinstance(segmentation, list):
            stats["num_polygon"] += 1
        else:
            raise ValueError(
                f"Unsupported segmentation type in annotation {annotation_id}: {type(segmentation)}"
            )

        if not disable_rle_validation:
            decoded_area = validate_rle_annotation(
                annotation, image["width"], image["height"]
            )

            reported_area = annotation.get("area")
            if decoded_area is not None and reported_area is not None:
                relative_error = abs(decoded_area - reported_area) / max(decoded_area, 1)

                if relative_error > 0.01:
                    print(f"WARNING: annotation {annotation_id} has area mismatch:")
                    print(f"    decoded : {decoded_area}")
                    print(f"    reported: {reported_area}")

        stats["num_annotations"] += 1

    print(f"  Validated {stats['num_annotations']} annotations.")
    print(f"  RLE masks     : {stats['num_rle']}")
    print(f"  Polygon masks : {stats['num_polygon']}")

    if mask_utils is None and not disable_rle_validation:
        print("\nWARNING: pycocotools is not installed. RLE masks were not decoded.")
        print("Install with: pip install pycocotools")


# ==========================================================
# COCO RECORD HELPERS
# ==========================================================

def sanitize_image_record(image):
    return {
        "id": image["id"],
        "file_name": image["file_name"],
        "width": image["width"],
        "height": image["height"],
    }


def sanitize_annotation_record(annotation):
    result = {
        "id": annotation["id"],
        "image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "bbox": annotation["bbox"],
        "area": annotation["area"],
        "iscrowd": annotation.get("iscrowd", 0),
        "segmentation": annotation["segmentation"],
    }

    if "ignore" in annotation:
        result["ignore"] = annotation["ignore"]

    return result


def create_split_json(source_data, image_records, annotation_records, description):
    info = source_data.get("info", {}).copy()
    info["description"] = description

    result = {
        "info": info,
        "licenses": source_data.get("licenses", []),
        "categories": source_data["categories"],
        "images": image_records,
        "annotations": annotation_records,
    }

    if "videos" in source_data:
        result["videos"] = source_data["videos"]

    return result


def write_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ==========================================================
# K-FOLD SPLITTING
# ==========================================================

def create_kfold_splits(image_ids, num_folds, seed):
    num_images = len(image_ids)

    if num_folds < 2:
        raise ValueError("--num_folds must be >= 2.")

    if num_folds > num_images:
        raise ValueError(
            f"--num_folds={num_folds} cannot be greater than the number of images ({num_images})."
        )

    shuffled_ids = list(image_ids)
    random.Random(seed).shuffle(shuffled_ids)

    base_size, remainder = divmod(num_images, num_folds)
    fold_sizes = [base_size + (i < remainder) for i in range(num_folds)]

    folds, current_index = [], 0

    for fold_index, fold_size in enumerate(fold_sizes):
        valid_ids = sorted(shuffled_ids[current_index:current_index + fold_size])
        valid_id_set = set(valid_ids)
        train_ids = sorted(image_id for image_id in image_ids if image_id not in valid_id_set)

        folds.append({
            "fold": fold_index + 1,
            "train_ids": train_ids,
            "valid_ids": valid_ids,
        })

        current_index += fold_size

    validation_ids = [image_id for fold in folds for image_id in fold["valid_ids"]]

    for fold in folds:
        if set(fold["train_ids"]) & set(fold["valid_ids"]):
            raise RuntimeError(f"Train/validation overlap in fold {fold['fold']}.")

    if set(validation_ids) != set(image_ids):
        raise RuntimeError(
            "K-fold split is invalid: validation sets do not cover every image exactly once."
        )

    if len(validation_ids) != len(set(validation_ids)):
        raise RuntimeError(
            "K-fold split is invalid: an image appears in validation more than once."
        )

    return folds


# ==========================================================
# CREATE ONE FOLD
# ==========================================================

def copy_split_images(split_name, image_records, resolved_paths, fold_dir):
    split_dir = fold_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for image in image_records:
        source_path = resolved_paths[image["id"]]
        destination_name = Path(image["file_name"]).name
        shutil.copy2(source_path, split_dir / destination_name)
        image["file_name"] = destination_name


def create_fold(fold_definition, source_data, images_by_id, annotations, resolved_paths, output_dir):
    fold_number = fold_definition["fold"]
    train_ids, valid_ids = fold_definition["train_ids"], fold_definition["valid_ids"]

    fold_dir = output_dir / f"fold_{fold_number:02d}"
    (fold_dir / "train").mkdir(parents=True, exist_ok=True)
    (fold_dir / "valid").mkdir(parents=True, exist_ok=True)

    train_images = [sanitize_image_record(images_by_id[i]) for i in train_ids]
    valid_images = [sanitize_image_record(images_by_id[i]) for i in valid_ids]

    train_id_set, valid_id_set = set(train_ids), set(valid_ids)

    train_annotations = [
        sanitize_annotation_record(a)
        for a in annotations
        if a["image_id"] in train_id_set
    ]

    valid_annotations = [
        sanitize_annotation_record(a)
        for a in annotations
        if a["image_id"] in valid_id_set
    ]

    copy_split_images("train", train_images, resolved_paths, fold_dir)
    copy_split_images("valid", valid_images, resolved_paths, fold_dir)

    train_json = create_split_json(
        source_data,
        train_images,
        train_annotations,
        f"RF-DETR K-fold training split - fold {fold_number}",
    )

    valid_json = create_split_json(
        source_data,
        valid_images,
        valid_annotations,
        f"RF-DETR K-fold validation split - fold {fold_number}",
    )

    train_json_path = fold_dir / "train" / "_annotations.coco.json"
    valid_json_path = fold_dir / "valid" / "_annotations.coco.json"

    write_json(train_json, train_json_path)
    write_json(valid_json, valid_json_path)

    manifest = {
        "fold": fold_number,
        "train_image_ids": train_ids,
        "valid_image_ids": valid_ids,
        "train_image_count": len(train_ids),
        "valid_image_count": len(valid_ids),
        "train_annotation_count": len(train_annotations),
        "valid_annotation_count": len(valid_annotations),
        "train_json": str(train_json_path),
        "valid_json": str(valid_json_path),
    }

    manifest_path = fold_dir / "split_manifest.json"
    write_json(manifest, manifest_path)

    print(f"\n{'-' * 60}")
    print(f"FOLD {fold_number}")
    print(f"  Train images      : {len(train_ids)}")
    print(f"  Validation images : {len(valid_ids)}")
    print(f"  Train annotations : {len(train_annotations)}")
    print(f"  Valid annotations : {len(valid_annotations)}")
    print(f"  Train IDs         : {train_ids}")
    print(f"  Valid IDs         : {valid_ids}")
    print(f"  Output            : {fold_dir}")

    return {
        "fold": fold_number,
        "train_image_ids": train_ids,
        "valid_image_ids": valid_ids,
        "train_image_count": len(train_ids),
        "valid_image_count": len(valid_ids),
        "train_annotation_count": len(train_annotations),
        "valid_annotation_count": len(valid_annotations),
        "path": str(fold_dir),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():
    args = parse_args()

    annotations_path = Path(args.annotations).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation file not found:\n{annotations_path}")

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found:\n{images_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty:\n{output_dir}\n\n"
            f"Use --overwrite to allow recreating the folds."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(annotations_path)
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    print("\nDataset summary:")
    print(f"  Images        : {len(images)}")
    print(f"  Annotations   : {len(annotations)}")
    print(f"  Categories    : {len(categories)}")
    print(f"  Requested folds: {args.num_folds}")

    category_ids = validate_categories(categories)
    images_by_id = build_image_index(images)
    resolved_paths = validate_images(images, images_dir)

    validate_annotations(
        annotations,
        images_by_id,
        category_ids,
        disable_rle_validation=args.no_rle_validation,
    )

    all_image_ids = sorted(images_by_id.keys())

    folds = create_kfold_splits(
        image_ids=all_image_ids,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION SPLITS")
    print("=" * 70)

    for fold in folds:
        print(
            f"Fold {fold['fold']:02d}: "
            f"train={len(fold['train_ids'])}, "
            f"valid={len(fold['valid_ids'])}"
        )

    fold_manifests = []

    for fold_definition in folds:
        fold_manifests.append(
            create_fold(
                fold_definition,
                data,
                images_by_id,
                annotations,
                resolved_paths,
                output_dir,
            )
        )

    global_manifest = {
        "source_annotations": str(annotations_path),
        "source_images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "num_folds": args.num_folds,
        "num_images": len(all_image_ids),
        "seed": args.seed,
        "image_ids": all_image_ids,
        "folds": fold_manifests,
    }

    global_manifest_path = output_dir / "cross_validation_manifest.json"
    write_json(global_manifest, global_manifest_path)

    print("\n" + "=" * 70)
    print("K-FOLD DATASET PREPARATION FINISHED")
    print("=" * 70)
    print(f"Total images : {len(all_image_ids)}")
    print(f"Number folds : {args.num_folds}")
    print(f"Output       : {output_dir}")
    print(f"Global manifest:\n  {global_manifest_path}")
    print("\nEach fold can now be passed independently to RF-DETR model.train().")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
