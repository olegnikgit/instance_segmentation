#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A simple example to demonstrate the use of custom trained RF-DETR model for object detection on an image.

@author: Olegs
"""

# ==========================================================
import gc
import torch
import weakref
from rfdetr import RFDETRNano
import supervision as sv
from PIL import Image
import os


# ==========================================================

def cleanup_gpu_memory(obj=None, verbose: bool = False):

    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA is not available. No GPU cleanup needed.")
        return

    def get_memory_stats():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[Before] Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")

    # Ensure we drop all strong references
    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully garbage collected yet.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[After]  Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")


def main():

    # clean up GPU memory before starting the inference
    cleanup_gpu_memory(verbose=True)

    home_dir = os.path.expanduser("~")
    pretrain_weights = os.path.join(home_dir, ".roboflow/models/rf-detr-nano.pth")

    model = RFDETRNano(resolution=384,
                       pretrain_weights=pretrain_weights)
    model.optimize_for_inference()

    # initialize the dataset using supervision's DetectionDataset.from_coco method:
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="DB_PATH",
        annotations_path="DB_PATH/_annotations.coco.json")

    path, image, annotations = ds[0]
    image = Image.open(path)

    detections = model.predict(image, threshold=0.5)

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    bbox_annotator = sv.BoxAnnotator(color=color,thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale)

    annotations_labels = [
        f"{ds.classes[class_id]}"
        for class_id
        in annotations.class_id
    ]

    detections_labels = [
        f"{ds.classes[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotation_image = image.copy()
    annotation_image = bbox_annotator.annotate(annotation_image, annotations)
    annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

    detections_image = image.copy()
    detections_image = bbox_annotator.annotate(detections_image, detections)
    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

    sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1, 2), titles=["Annotation", "Detection"])


# ==========================================================
if __name__ == "__main__":
    main()


