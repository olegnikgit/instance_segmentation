#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collection of evaluation utils for an object detection task.
"""

# ==========================================================
def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    The boxes are represented as [x, y, w, h], where (x, y) is the top-left corner
    and (w, h) are the width and height.

    Parameters:
    - boxA: [x, y, w, h]
    - boxB: [x, y, w, h]

    Returns:
    - iou: float
    """

    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    areaA = aw * ah
    areaB = bw * bh

    union = areaA + areaB - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union



