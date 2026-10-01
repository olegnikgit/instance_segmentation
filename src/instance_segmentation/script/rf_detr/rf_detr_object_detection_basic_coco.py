#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An example how to use RF-DETR for object detection.

@author: Olegs
"""

# ==========================================================
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
import pkg_resources


# ==========================================================


def main():

    # read the image located in the package: src/instance_segmentation/test/data/dog-2.jpeg:
    test_file = pkg_resources.resource_filename("instance_segmentation", "test/data/dog-2.jpeg")

    image = Image.open(test_file)

    model = RFDETRMedium(resolution=640)
    model.optimize_for_inference()

    detections = model.predict(image, threshold=0.5)

    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True
    )

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    annotated_image.show()

# ==========================================================
if __name__ == "__main__":
    main()


