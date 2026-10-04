#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate N-fold RF-DETR instance segmentation.

For every fold:

    dataset_root/
        fold_XX/
            valid/
                _annotations.coco.json

    cv_dir/
        fold_XX/
            checkpoint_best_total.pth
            predictions/
                predictions.json

Reports:

    Localization:
        Precision @ bbox IoU 0.50
        Recall    @ bbox IoU 0.50

    Segmentation:
        Mean mask IoU

    COCO instance segmentation:
        Mask AP50
        Mask AP75
        Mask AP50:95

Also calculates:
    - mean +/- standard deviation over folds
    - min/max over folds
    - pooled out-of-fold metrics

Example:

python evaluate_kfold.py \
    --cv_dir /path/to/experiments/rf_detr_5fold \
    --dataset_root /path/to/dataset/rf_detr_5fold \
    --prediction_name predictions.json \
    --confidence 0.5 \
    --iou_threshold 0.5
"""

import argparse
import csv
import copy
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate K-fold RF-DETR instance segmentation."
    )

    parser.add_argument(
        "--cv_dir",
        type=str,
        required=True,
        help="Directory containing fold_01, fold_02, ... with predictions.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Directory containing fold_01, fold_02, ... with ground truth.",
    )

    parser.add_argument(
        "--prediction_name",
        type=str,
        default="predictions.json",
        help="Prediction JSON filename.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for precision/recall.",
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="BBox IoU threshold for localization matching.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results.",
    )

    return parser.parse_args()


# ==========================================================
# JSON
# ==========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# MASK HELPERS
# ==========================================================

def decode_rle(segmentation):
    mask = mask_utils.decode(segmentation)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(bool)


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


# ==========================================================
# BBOX HELPERS
# ==========================================================

def xywh_to_xyxy(bbox):
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y + h]


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ==========================================================
# FIXED-THRESHOLD MATCHING
# ==========================================================

def match_predictions(
    gt_annotations,
    prediction_annotations,
    confidence_threshold,
    iou_threshold,
):
    """
    One-to-one greedy matching.

    Predictions are processed by descending confidence.

    A prediction matches a GT if:
        - same category_id
        - bbox IoU >= iou_threshold
    """

    predictions = [
        prediction
        for prediction in prediction_annotations
        if float(prediction.get("confidence", prediction.get("score", 0.0)))
        >= confidence_threshold
    ]

    predictions.sort(
        key=lambda x: float(
            x.get("confidence", x.get("score", 0.0))
        ),
        reverse=True,
    )

    gt_unused = set(range(len(gt_annotations)))

    matches = []
    unmatched_predictions = []

    for prediction in predictions:
        best_gt = None
        best_iou = 0.0

        for gt_index in gt_unused:
            gt = gt_annotations[gt_index]

            if int(prediction["category_id"]) != int(gt["category_id"]):
                continue

            iou = bbox_iou(prediction["bbox"], gt["bbox"])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if best_gt is None:
            unmatched_predictions.append(prediction)
        else:
            gt_unused.remove(best_gt)
            matches.append(
                {
                    "prediction": prediction,
                    "ground_truth": gt_annotations[best_gt],
                    "bbox_iou": best_iou,
                }
            )

    unmatched_ground_truth = [
        gt_annotations[index]
        for index in sorted(gt_unused)
    ]

    return matches, unmatched_predictions, unmatched_ground_truth


# ==========================================================
# FIXED P/R + MASK IoU
# ==========================================================

def calculate_fixed_metrics(
    gt_data,
    prediction_data,
    confidence_threshold,
    iou_threshold,
):
    gt_by_image = {}
    pred_by_image = {}

    for annotation in gt_data["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)

    for annotation in prediction_data["annotations"]:
        pred_by_image.setdefault(annotation["image_id"], []).append(annotation)

    image_ids = sorted(set(gt_by_image) | set(pred_by_image))

    tp = 0
    fp = 0
    fn = 0
    mask_ious = []

    for image_id in image_ids:
        gt_annotations = gt_by_image.get(image_id, [])
        prediction_annotations = pred_by_image.get(image_id, [])

        matches, unmatched_predictions, unmatched_ground_truth = match_predictions(
            gt_annotations,
            prediction_annotations,
            confidence_threshold,
            iou_threshold,
        )

        tp += len(matches)
        fp += len(unmatched_predictions)
        fn += len(unmatched_ground_truth)

        for match in matches:
            gt_segmentation = match["ground_truth"].get("segmentation")
            pred_segmentation = match["prediction"].get("segmentation")

            if gt_segmentation is None or pred_segmentation is None:
                continue

            gt_mask = decode_rle(gt_segmentation)
            pred_mask = decode_rle(pred_segmentation)

            mask_ious.append(mask_iou(gt_mask, pred_mask))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    mean_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mask_iou": mean_mask_iou,
        "matched_instances": len(mask_ious),
    }


# ==========================================================
# COCO METRICS
# ==========================================================

def calculate_coco_segmentation_metrics(gt_data, prediction_data):
    """
    Calculate standard COCO segmentation metrics.

    Returns:
        AP50:95
        AP50
        AP75

    Important:
        COCOeval.summarize() must be called after accumulate()
        before reading evaluator.stats.
    """

    # No predictions -> AP = 0.
    if not prediction_data.get("annotations"):
        return {
            "mask_ap50_95": 0.0,
            "mask_ap50": 0.0,
            "mask_ap75": 0.0,
        }

    coco_gt = COCO()
    coco_gt.dataset = gt_data
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(prediction_data["annotations"])

    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.imgIds = [image["id"] for image in gt_data["images"]]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()   # <-- required before evaluator.stats

    stats = evaluator.stats

    return {
        "mask_ap50_95": float(stats[0]),
        "mask_ap50": float(stats[1]),
        "mask_ap75": float(stats[2]),
    }


# ==========================================================
# EVALUATE ONE FOLD
# ==========================================================

def evaluate_fold(
    fold_dir,
    dataset_fold_dir,
    prediction_name,
    confidence_threshold,
    iou_threshold,
):
    gt_path = dataset_fold_dir / "valid" / "_annotations.coco.json"

    prediction_path = fold_dir / "predictions" / prediction_name

    if not prediction_path.is_file():
        prediction_path = fold_dir / prediction_name

    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found:\n{gt_path}")

    if not prediction_path.is_file():
        raise FileNotFoundError(f"Predictions not found:\n{prediction_path}")

    gt_data = load_json(gt_path)
    prediction_data = load_json(prediction_path)

    fixed = calculate_fixed_metrics(
        gt_data,
        prediction_data,
        confidence_threshold,
        iou_threshold,
    )

    coco = calculate_coco_segmentation_metrics(
        gt_data,
        prediction_data,
    )

    return {
        **fixed,
        **coco,
        "validation_images": len(gt_data["images"]),
        "ground_truth_instances": len(gt_data["annotations"]),
        "predicted_instances": len(prediction_data["annotations"]),
    }


# ==========================================================
# STATISTICS
# ==========================================================

def mean_std(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0, 0.0

    if len(values) == 1:
        return float(values[0]), 0.0

    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize_folds(fold_metrics):
    metric_names = [
        "precision",
        "recall",
        "mask_iou",
        "mask_ap50",
        "mask_ap75",
        "mask_ap50_95",
    ]

    summary = {}

    for metric in metric_names:
        values = [fold[metric] for fold in fold_metrics]
        mean, std = mean_std(values)

        summary[metric] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return summary


# ==========================================================
# POOLED OOF EVALUATION
# ==========================================================

def pool_oof_predictions(fold_dirs, prediction_name):
    pooled = None

    for fold_dir in fold_dirs:
        prediction_path = fold_dir / "predictions" / prediction_name

        if not prediction_path.is_file():
            prediction_path = fold_dir / prediction_name

        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_path}"
            )

        data = load_json(prediction_path)

        if pooled is None:
            pooled = copy.deepcopy(data)
            pooled["images"] = []
            pooled["annotations"] = []

        pooled["images"].extend(data["images"])
        pooled["annotations"].extend(data["annotations"])

    image_by_id = {image["id"]: image for image in pooled["images"]}
    pooled["images"] = list(image_by_id.values())

    return pooled


def build_pooled_gt(fold_dirs, dataset_root):
    first_gt = None
    images = []
    annotations = []
    seen_image_ids = set()

    for fold_dir in fold_dirs:
        gt_path = (
            dataset_root
            / fold_dir.name
            / "valid"
            / "_annotations.coco.json"
        )

        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Ground truth not found: {gt_path}"
            )

        data = load_json(gt_path)

        if first_gt is None:
            first_gt = copy.deepcopy(data)
            first_gt["images"] = []
            first_gt["annotations"] = []

        for image in data["images"]:
            if image["id"] in seen_image_ids:
                raise RuntimeError(
                    f"Image ID {image['id']} appears in more than one validation fold."
                )

            seen_image_ids.add(image["id"])
            images.append(image)

        annotations.extend(data["annotations"])

    first_gt["images"] = images
    first_gt["annotations"] = annotations

    return first_gt


# ==========================================================
# CSV
# ==========================================================

def write_csv(path, fold_metrics):
    fields = [
        "fold",
        "validation_images",
        "ground_truth_instances",
        "predicted_instances",
        "precision",
        "recall",
        "mask_iou",
        "mask_ap50",
        "mask_ap75",
        "mask_ap50_95",
        "tp",
        "fp",
        "fn",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for fold in fold_metrics:
            writer.writerow(
                {field: fold.get(field, "") for field in fields}
            )


# ==========================================================
# MAIN
# ==========================================================

def main():
    args = parse_args()

    cv_dir = Path(args.cv_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if not cv_dir.is_dir():
        raise FileNotFoundError(f"CV directory not found:\n{cv_dir}")

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found:\n{dataset_root}")

    if args.output_dir is None:
        output_dir = cv_dir / "evaluation"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(
        path
        for path in cv_dir.glob("fold_*")
        if path.is_dir()
    )

    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {cv_dir}")

    print()
    print("=" * 80)
    print("RF-DETR K-FOLD EVALUATION")
    print("=" * 80)
    print(f"CV directory       : {cv_dir}")
    print(f"Dataset root       : {dataset_root}")
    print(f"Number of folds    : {len(fold_dirs)}")
    print(f"Confidence          : {args.confidence}")
    print(f"BBox IoU threshold  : {args.iou_threshold}")
    print("=" * 80)

    # ------------------------------------------------------
    # Evaluate every fold
    # ------------------------------------------------------

    fold_metrics = []

    for fold_dir in fold_dirs:
        print()
        print("=" * 80)
        print(f"EVALUATING {fold_dir.name}")
        print("=" * 80)

        metrics = evaluate_fold(
            fold_dir=fold_dir,
            dataset_fold_dir=dataset_root / fold_dir.name,
            prediction_name=args.prediction_name,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
        )

        metrics["fold"] = int(fold_dir.name.split("_")[-1])
        fold_metrics.append(metrics)

        print(
            f"Precision @ IoU {args.iou_threshold:.2f}: "
            f"{metrics['precision']:.4f}"
        )
        print(
            f"Recall @ IoU {args.iou_threshold:.2f}:    "
            f"{metrics['recall']:.4f}"
        )
        print(f"Mask IoU:             {metrics['mask_iou']:.4f}")
        print(f"Mask AP50:            {metrics['mask_ap50']:.4f}")
        print(f"Mask AP75:            {metrics['mask_ap75']:.4f}")
        print(f"Mask AP50:95:         {metrics['mask_ap50_95']:.4f}")

    fold_metrics.sort(key=lambda x: x["fold"])

    # ------------------------------------------------------
    # Per-fold CSV
    # ------------------------------------------------------

    csv_path = output_dir / "fold_metrics.csv"
    write_csv(csv_path, fold_metrics)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = summarize_folds(fold_metrics)

    print()
    print("=" * 100)
    print("PER-FOLD RESULTS")
    print("=" * 100)

    header = (
        f"{'Fold':<6}"
        f"{'Images':<9}"
        f"{'Prec@.50':<12}"
        f"{'Recall@.50':<12}"
        f"{'Mask IoU':<12}"
        f"{'AP50':<12}"
        f"{'AP75':<12}"
        f"{'AP50:95':<12}"
    )

    print(header)
    print("-" * len(header))

    for fold in fold_metrics:
        print(
            f"{fold['fold']:<6}"
            f"{fold['validation_images']:<9}"
            f"{fold['precision']:<12.4f}"
            f"{fold['recall']:<12.4f}"
            f"{fold['mask_iou']:<12.4f}"
            f"{fold['mask_ap50']:<12.4f}"
            f"{fold['mask_ap75']:<12.4f}"
            f"{fold['mask_ap50_95']:<12.4f}"
        )

    print("-" * len(header))

    def fmt(metric):
        return (
            f"{summary[metric]['mean']:.4f} "
            f"+/- "
            f"{summary[metric]['std']:.4f}"
        )

    print(
        f"{'Mean ± SD':<15}"
        f"{'':<9}"
        f"{fmt('precision'):<12}"
        f"{fmt('recall'):<12}"
        f"{fmt('mask_iou'):<12}"
        f"{fmt('mask_ap50'):<12}"
        f"{fmt('mask_ap75'):<12}"
        f"{fmt('mask_ap50_95'):<12}"
    )

    # ------------------------------------------------------
    # Pooled out-of-fold evaluation
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 80)

    pooled_gt = build_pooled_gt(fold_dirs, dataset_root)

    pooled_predictions = pool_oof_predictions(
        fold_dirs,
        args.prediction_name,
    )

    pooled_fixed = calculate_fixed_metrics(
        pooled_gt,
        pooled_predictions,
        args.confidence,
        args.iou_threshold,
    )

    pooled_coco = calculate_coco_segmentation_metrics(
        pooled_gt,
        pooled_predictions,
    )

    pooled = {
        **pooled_fixed,
        **pooled_coco,
        "images": len(pooled_gt["images"]),
        "ground_truth_instances": len(pooled_gt["annotations"]),
        "predicted_instances": len(pooled_predictions["annotations"]),
    }

    print(
        f"Precision @ IoU {args.iou_threshold:.2f}: "
        f"{pooled['precision']:.4f}"
    )
    print(
        f"Recall @ IoU {args.iou_threshold:.2f}:    "
        f"{pooled['recall']:.4f}"
    )
    print(f"Mask IoU:             {pooled['mask_iou']:.4f}")
    print(f"Mask AP50:            {pooled['mask_ap50']:.4f}")
    print(f"Mask AP75:            {pooled['mask_ap75']:.4f}")
    print(f"Mask AP50:95:         {pooled['mask_ap50_95']:.4f}")

    # ------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------

    results = {
        "evaluation": {
            "confidence_threshold": args.confidence,
            "bbox_iou_threshold": args.iou_threshold,
        },
        "folds": fold_metrics,
        "mean_std": summary,
        "pooled_out_of_fold": pooled,
    }

    summary_path = output_dir / "cv_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("EVALUATION FINISHED")
    print("=" * 80)
    print(f"Per-fold CSV : {csv_path}")
    print(f"Summary JSON : {summary_path}")


if __name__ == "__main__":
    main()
