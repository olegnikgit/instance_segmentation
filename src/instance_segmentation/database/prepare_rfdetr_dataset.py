#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a small COCO-style instance-segmentation dataset for RF-DETR.

Input:
    - One COCO-like annotations.json
    - A directory containing the images referenced by the JSON

Output:
    dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── image1.jpg
    │   └── ...
    └── valid/
        ├── _annotations.coco.json
        └── ...

The script:
    1. Reads the original annotations.json.
    2. Validates the referenced images.
    3. Validates image dimensions.
    4. Validates segmentation RLE masks when possible.
    5. Splits images into train/valid.
    6. Copies the images into the RF-DETR directory structure.
    7. Writes clean COCO annotation files.
    8. Preserves instance segmentation masks as RLE.
    9. Writes a split manifest for reproducibility.

Example:

python prepare_rfdetr_dataset.py \
    --annotations /path/to/annotations.json \
    --images_dir /path/to/images \
    --output_dir /path/to/rfdetr_dataset \
    --val_count 1 \
    --seed 42

To explicitly select the validation image:

python prepare_rfdetr_dataset.py \
    --annotations /path/to/annotations.json \
    --images_dir /path/to/images \
    --output_dir /path/to/rfdetr_dataset \
    --val_image_id 269

@author: Olegs
"""

# ==========================================================
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small COCO instance-segmentation dataset "
            "for RF-DETR."
        )
    )

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the original annotations.json file.",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help=(
            "Directory containing the images referenced by "
            "the annotation JSON."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output RF-DETR dataset directory.",
    )

    parser.add_argument(
        "--val_count",
        type=int,
        default=1,
        help=(
            "Number of images to put into validation. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--val_image_id",
        type=int,
        default=None,
        help=(
            "Optional explicit image ID to use for validation. "
            "If provided, --val_count is ignored."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )

    parser.add_argument(
        "--copy_images",
        action="store_true",
        default=True,
        help="Copy images into train/valid directories.",
    )

    parser.add_argument(
        "--no_rle_validation",
        action="store_true",
        help=(
            "Do not decode and validate segmentation masks. "
            "Useful only if pycocotools is unavailable."
        ),
    )

    return parser.parse_args()


# ==========================================================
def load_json(path: Path):
    print(f"Loading annotations: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["images", "annotations", "categories"]

    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"Input JSON does not contain required key: '{key}'"
            )

    return data


# ==========================================================
def validate_categories(categories):
    if not categories:
        raise ValueError("No categories found in annotations.")

    category_ids = set()

    print("\nCategories:")

    for category in categories:
        category_id = category["id"]
        category_name = category["name"]

        if category_id in category_ids:
            raise ValueError(
                f"Duplicate category ID: {category_id}"
            )

        category_ids.add(category_id)

        print(
            f"  id={category_id}: {category_name}"
        )

    return category_ids


# ==========================================================
def build_annotation_index(annotations):
    """
    Map image_id -> annotations.
    """
    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        image_id = annotation["image_id"]
        annotations_by_image[image_id].append(annotation)

    return annotations_by_image


# ==========================================================
def resolve_image_path(images_dir: Path, file_name: str):
    """
    Resolve image path.

    First try:
        images_dir / file_name

    If that does not exist, also try basename.
    """
    path = images_dir / file_name

    if path.exists():
        return path

    basename_path = images_dir / Path(file_name).name

    if basename_path.exists():
        return basename_path

    return None


# ==========================================================
def validate_images(
    images,
    images_dir: Path,
):
    """
    Check that all referenced images exist and that
    their dimensions match the annotation JSON.
    """
    print("\nValidating images...")

    resolved_paths = {}

    for image in images:
        image_id = image["id"]
        file_name = image["file_name"]
        expected_width = image["width"]
        expected_height = image["height"]

        image_path = resolve_image_path(
            images_dir,
            file_name,
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Image not found:\n"
                f"  image_id : {image_id}\n"
                f"  file_name: {file_name}\n"
                f"  searched : {images_dir}"
            )

        try:
            with Image.open(image_path) as pil_image:
                actual_width, actual_height = pil_image.size

        except Exception as exc:
            raise RuntimeError(
                f"Could not open image '{image_path}': {exc}"
            ) from exc

        if (
            actual_width != expected_width
            or actual_height != expected_height
        ):
            raise ValueError(
                f"Image dimension mismatch for '{file_name}':\n"
                f"  JSON : {expected_width} x {expected_height}\n"
                f"  file : {actual_width} x {actual_height}"
            )

        resolved_paths[image_id] = image_path

    print(
        f"  Validated {len(images)} images successfully."
    )

    return resolved_paths


# ==========================================================
def validate_rle_annotation(
    annotation,
    image_width,
    image_height,
):
    """
    Validate a COCO RLE segmentation.

    Supports:
        - compressed RLE
        - uncompressed RLE

    Returns:
        decoded mask area when possible
    """
    segmentation = annotation.get("segmentation")

    if segmentation is None:
        raise ValueError(
            f"Annotation {annotation['id']} has no segmentation."
        )

    # ------------------------------------------------------
    # Polygon segmentation
    #
    # Not expected for your current dataset, but don't fail
    # just because a polygon appears.
    # ------------------------------------------------------
    if isinstance(segmentation, list):
        if mask_utils is None:
            return None

        rles = mask_utils.frPyObjects(
            segmentation,
            image_height,
            image_width,
        )

        decoded = mask_utils.decode(rles)

        if decoded.ndim == 3:
            decoded_area = int(decoded.sum())
        else:
            decoded_area = int(decoded.sum())

        return decoded_area

    # ------------------------------------------------------
    # RLE segmentation
    # ------------------------------------------------------
    if not isinstance(segmentation, dict):
        raise ValueError(
            f"Annotation {annotation['id']} has unsupported "
            f"segmentation type: {type(segmentation)}"
        )

    rle_height, rle_width = segmentation["size"]

    if (
        rle_height != image_height
        or rle_width != image_width
    ):
        raise ValueError(
            f"RLE size mismatch in annotation "
            f"{annotation['id']}:\n"
            f"  RLE    : {rle_width} x {rle_height}\n"
            f"  image  : {image_width} x {image_height}"
        )

    # If pycocotools is unavailable, size validation above
    # is still useful.
    if mask_utils is None:
        return None

    try:
        decoded = mask_utils.decode(segmentation)
    except Exception as exc:
        raise ValueError(
            f"Could not decode RLE for annotation "
            f"{annotation['id']}: {exc}"
        ) from exc

    decoded_area = int(decoded.sum())

    if decoded_area <= 0:
        raise ValueError(
            f"Annotation {annotation['id']} has an empty mask."
        )

    return decoded_area


# ==========================================================
def validate_annotations(
    annotations,
    images_by_id,
    category_ids,
    disable_rle_validation=False,
):
    """
    Validate annotations and segmentation masks.
    """
    print("\nValidating annotations...")

    annotation_ids = set()

    stats = {
        "num_annotations": 0,
        "num_rle": 0,
        "num_polygon": 0,
    }

    for annotation in annotations:

        annotation_id = annotation["id"]

        if annotation_id in annotation_ids:
            raise ValueError(
                f"Duplicate annotation ID: {annotation_id}"
            )

        annotation_ids.add(annotation_id)

        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in images_by_id:
            raise ValueError(
                f"Annotation {annotation_id} references "
                f"unknown image_id={image_id}"
            )

        if category_id not in category_ids:
            raise ValueError(
                f"Annotation {annotation_id} references "
                f"unknown category_id={category_id}"
            )

        image = images_by_id[image_id]

        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox in annotation {annotation_id}: "
                f"{bbox}"
            )

        x, y, width, height = bbox

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid bbox dimensions in annotation "
                f"{annotation_id}: {bbox}"
            )

        segmentation = annotation.get("segmentation")

        if segmentation is None:
            raise ValueError(
                f"Annotation {annotation_id} has no segmentation."
            )

        if isinstance(segmentation, dict):
            stats["num_rle"] += 1

        elif isinstance(segmentation, list):
            stats["num_polygon"] += 1

        else:
            raise ValueError(
                f"Unsupported segmentation type in "
                f"annotation {annotation_id}: "
                f"{type(segmentation)}"
            )

        if not disable_rle_validation:

            decoded_area = validate_rle_annotation(
                annotation=annotation,
                image_width=image["width"],
                image_height=image["height"],
            )

            # Check reported area when we were able to decode
            # the mask.
            if decoded_area is not None:
                reported_area = annotation.get("area")

                if reported_area is not None:
                    relative_error = abs(
                        decoded_area - reported_area
                    ) / max(decoded_area, 1)

                    if relative_error > 0.01:
                        print(
                            f"WARNING: annotation "
                            f"{annotation_id} has area mismatch:"
                        )
                        print(
                            f"    decoded : {decoded_area}"
                        )
                        print(
                            f"    reported: {reported_area}"
                        )

        stats["num_annotations"] += 1

    print(
        f"  Validated {stats['num_annotations']} annotations."
    )
    print(
        f"  RLE masks     : {stats['num_rle']}"
    )
    print(
        f"  Polygon masks : {stats['num_polygon']}"
    )

    if mask_utils is None and not disable_rle_validation:
        print(
            "\nWARNING: pycocotools is not installed. "
            "RLE masks were not decoded."
        )
        print(
            "Install with: pip install pycocotools"
        )


# ==========================================================
def split_image_ids(
    image_ids,
    val_count,
    seed,
    val_image_id=None,
):
    """
    Deterministically create train/validation split.
    """
    image_ids = list(image_ids)

    if len(image_ids) < 2:
        raise ValueError(
            "At least two images are required for a "
            "train/validation split."
        )

    if val_image_id is not None:

        if val_image_id not in image_ids:
            raise ValueError(
                f"--val_image_id={val_image_id} does not exist."
            )

        valid_ids = [val_image_id]

    else:

        if val_count < 1:
            raise ValueError(
                "--val_count must be at least 1."
            )

        if val_count >= len(image_ids):
            raise ValueError(
                "Validation set must contain fewer images "
                "than the complete dataset."
            )

        rng = random.Random(seed)

        shuffled = image_ids.copy()
        rng.shuffle(shuffled)

        valid_ids = sorted(
            shuffled[:val_count]
        )

    valid_id_set = set(valid_ids)

    train_ids = sorted(
        image_id
        for image_id in image_ids
        if image_id not in valid_id_set
    )

    valid_ids = sorted(valid_ids)

    if not train_ids:
        raise ValueError(
            "Training split is empty."
        )

    return train_ids, valid_ids


# ==========================================================
def sanitize_image_record(image):
    """
    Create a clean COCO image entry.

    Keep only fields RF-DETR needs.
    """
    return {
        "id": image["id"],
        "file_name": image["file_name"],
        "width": image["width"],
        "height": image["height"],
    }


# ==========================================================
def sanitize_annotation_record(annotation):
    """
    Create a clean COCO instance-segmentation annotation.

    Preserve:
        - id
        - image_id
        - category_id
        - bbox
        - area
        - iscrowd
        - segmentation

    Preserve "ignore" when available.
    """
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


# ==========================================================
def create_split_json(
    source_data,
    image_records,
    annotation_records,
    description,
):
    """
    Build a clean COCO JSON object.
    """
    info = source_data.get(
        "info",
        {},
    ).copy()

    info["description"] = description

    return {
        "info": info,
        "licenses": source_data.get(
            "licenses",
            [],
        ),
        "categories": source_data["categories"],
        "images": image_records,
        "annotations": annotation_records,
    }


# ==========================================================
def copy_split_images(
    split_name,
    image_records,
    resolved_paths,
    output_dir,
):
    """
    Copy images into the split directory.

    RF-DETR expects the image paths referenced by
    file_name to exist relative to the split directory.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image in image_records:

        image_id = image["id"]

        source_path = resolved_paths[image_id]

        destination_name = Path(
            image["file_name"]
        ).name

        destination_path = (
            split_dir / destination_name
        )

        if (
            source_path.resolve()
            != destination_path.resolve()
        ):
            shutil.copy2(
                source_path,
                destination_path,
            )

        # Ensure the COCO file points to exactly the
        # copied filename.
        image["file_name"] = destination_name


# ==========================================================
def write_json(data, path: Path):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")


# ==========================================================
def main():

    args = parse_args()

    annotations_path = Path(
        args.annotations
    ).expanduser().resolve()

    images_dir = Path(
        args.images_dir
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    # ------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: "
            f"{annotations_path}"
        )

    if not images_dir.is_dir():
        raise NotADirectoryError(
            f"Images directory not found: "
            f"{images_dir}"
        )

    # ------------------------------------------------------
    # Load original annotations
    # ------------------------------------------------------
    data = load_json(
        annotations_path
    )

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    print("\nDataset summary:")
    print(
        f"  Images      : {len(images)}"
    )
    print(
        f"  Annotations : {len(annotations)}"
    )
    print(
        f"  Categories  : {len(categories)}"
    )

    # ------------------------------------------------------
    # Validate categories
    # ------------------------------------------------------
    category_ids = validate_categories(
        categories
    )

    # ------------------------------------------------------
    # Index images and annotations
    # ------------------------------------------------------
    images_by_id = {
        image["id"]: image
        for image in images
    }

    annotation_by_image = (
        build_annotation_index(
            annotations
        )
    )

    # ------------------------------------------------------
    # Validate image IDs
    # ------------------------------------------------------
    if len(images_by_id) != len(images):
        raise ValueError(
            "Duplicate image IDs detected."
        )

    # ------------------------------------------------------
    # Validate images
    # ------------------------------------------------------
    resolved_paths = validate_images(
        images=images,
        images_dir=images_dir,
    )

    # ------------------------------------------------------
    # Validate annotations / RLE
    # ------------------------------------------------------
    validate_annotations(
        annotations=annotations,
        images_by_id=images_by_id,
        category_ids=category_ids,
        disable_rle_validation=args.no_rle_validation,
    )

    # ------------------------------------------------------
    # Create train/valid split
    # ------------------------------------------------------
    all_image_ids = sorted(
        images_by_id.keys()
    )

    train_ids, valid_ids = split_image_ids(
        image_ids=all_image_ids,
        val_count=args.val_count,
        seed=args.seed,
        val_image_id=args.val_image_id,
    )

    print("\nDataset split:")
    print(
        f"  Train: {len(train_ids)} images"
    )
    print(
        f"  Valid: {len(valid_ids)} images"
    )

    print(
        f"\n  Train image IDs: {train_ids}"
    )
    print(
        f"  Valid image IDs: {valid_ids}"
    )

    # ------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"

    train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Prepare image records
    # ------------------------------------------------------
    train_images = [
        sanitize_image_record(
            images_by_id[image_id]
        )
        for image_id in train_ids
    ]

    valid_images = [
        sanitize_image_record(
            images_by_id[image_id]
        )
        for image_id in valid_ids
    ]

    # ------------------------------------------------------
    # Prepare annotation records
    # ------------------------------------------------------
    train_id_set = set(train_ids)
    valid_id_set = set(valid_ids)

    train_annotations = [
        sanitize_annotation_record(annotation)
        for annotation in annotations
        if annotation["image_id"] in train_id_set
    ]

    valid_annotations = [
        sanitize_annotation_record(annotation)
        for annotation in annotations
        if annotation["image_id"] in valid_id_set
    ]

    # ------------------------------------------------------
    # Copy images
    # ------------------------------------------------------
    copy_split_images(
        split_name="train",
        image_records=train_images,
        resolved_paths=resolved_paths,
        output_dir=output_dir,
    )

    copy_split_images(
        split_name="valid",
        image_records=valid_images,
        resolved_paths=resolved_paths,
        output_dir=output_dir,
    )

    # ------------------------------------------------------
    # Build clean COCO JSON
    # ------------------------------------------------------
    train_json = create_split_json(
        source_data=data,
        image_records=train_images,
        annotation_records=train_annotations,
        description=(
            "RF-DETR instance segmentation training split"
        ),
    )

    valid_json = create_split_json(
        source_data=data,
        image_records=valid_images,
        annotation_records=valid_annotations,
        description=(
            "RF-DETR instance segmentation validation split"
        ),
    )

    # ------------------------------------------------------
    # Write annotation files
    # ------------------------------------------------------
    train_json_path = (
        train_dir / "_annotations.coco.json"
    )

    valid_json_path = (
        valid_dir / "_annotations.coco.json"
    )

    write_json(
        train_json,
        train_json_path,
    )

    write_json(
        valid_json,
        valid_json_path,
    )

    # ------------------------------------------------------
    # Write a manifest so the exact split is reproducible
    # ------------------------------------------------------
    manifest = {
        "source_annotations": str(
            annotations_path
        ),
        "source_images_dir": str(
            images_dir
        ),
        "random_seed": args.seed,
        "train_image_ids": train_ids,
        "valid_image_ids": valid_ids,
        "train_image_count": len(train_ids),
        "valid_image_count": len(valid_ids),
    }

    manifest_path = (
        output_dir / "split_manifest.json"
    )

    write_json(
        manifest,
        manifest_path,
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print("RF-DETR dataset prepared successfully.")
    print("=" * 60)

    print(
        f"\nOutput directory:\n  {output_dir}"
    )

    print("\nGenerated files:")

    print(
        f"  {train_json_path}"
    )

    print(
        f"  {valid_json_path}"
    )

    print(
        f"  {manifest_path}"
    )

    print("\nDataset structure:")

    print(
        f"""
    {output_dir}/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── {len(train_images)} images
    └── valid/
        ├── _annotations.coco.json
        └── {len(valid_images)} images
    """
    )

    print(
        "You can now pass the output directory directly to "
        "RF-DETR model.train(dataset_dir=...)."
    )


# ==========================================================
if __name__ == "__main__":
    main()