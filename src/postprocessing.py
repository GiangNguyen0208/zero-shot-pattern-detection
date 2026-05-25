"""Post-processing: NMS, merging, and refinement of detections."""

import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two boxes in (x, y, w, h) format."""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(detections, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to remove overlapping detections.

    Keeps the detection with highest confidence when boxes overlap.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)

        remaining = []
        for det in sorted_dets:
            if compute_iou(best, det) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep


def merge_detections(template_dets, feature_dets, iou_threshold=0.3):
    """Merge detections from template matching and feature matching.

    If a feature detection overlaps with a template detection, keep the one
    with higher confidence. Feature-only detections are added if they have
    reasonable confidence.
    """
    all_dets = template_dets.copy()

    for fd in feature_dets:
        overlaps = False
        for td in all_dets:
            if compute_iou(fd, td) > iou_threshold:
                overlaps = True
                break
        if not overlaps and fd["confidence"] >= 0.3:
            all_dets.append(fd)

    return non_max_suppression(all_dets, iou_threshold)


def filter_by_aspect_ratio(detections, pattern_shape, tolerance=0.5):
    """Filter detections whose aspect ratio deviates too much from the pattern."""
    if not detections:
        return detections

    ph, pw = pattern_shape[:2]
    if ph == 0 or pw == 0:
        return detections

    pattern_ar = pw / ph
    filtered = []

    for det in detections:
        if det["h"] == 0:
            continue
        det_ar = det["w"] / det["h"]
        ratio = det_ar / pattern_ar if pattern_ar > 0 else 1.0
        if (1.0 - tolerance) <= ratio <= (1.0 + tolerance):
            filtered.append(det)

    return filtered


def clip_to_image(detections, img_shape):
    """Clip bounding boxes to image boundaries."""
    h, w = img_shape[:2]
    clipped = []
    for det in detections:
        d = det.copy()
        d["x"] = max(0, d["x"])
        d["y"] = max(0, d["y"])
        d["w"] = min(d["w"], w - d["x"])
        d["h"] = min(d["h"], h - d["y"])
        if d["w"] > 0 and d["h"] > 0:
            clipped.append(d)
    return clipped
