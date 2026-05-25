"""Visualization utilities for drawing bounding boxes on images."""

import cv2
import numpy as np


def draw_detections(image, detections, color=(0, 0, 255), thickness=3, font_scale=0.6):
    """Draw bounding boxes and confidence scores on image.

    Args:
        image: Input image (BGR or grayscale).
        detections: List of detection dicts with x, y, w, h, confidence.
        color: BGR color for bounding boxes.
        thickness: Line thickness.
        font_scale: Font scale for text.

    Returns:
        Image with drawn detections (BGR).
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, det in enumerate(detections):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        conf = det["confidence"]

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        label = f"#{i+1} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        label_y = y - 8 if y - 8 > th else y + h + th + 8
        label_x = x

        cv2.rectangle(vis, (label_x, label_y - th - 4), (label_x + tw + 4, label_y + 4), color, -1)
        cv2.putText(vis, label, (label_x + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def create_side_by_side(pattern, drawing_with_boxes, max_height=800):
    """Create a side-by-side visualization with pattern on left, result on right."""
    if len(pattern.shape) == 2:
        pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    else:
        pattern_bgr = pattern.copy()

    dh, dw = drawing_with_boxes.shape[:2]
    ph, pw = pattern_bgr.shape[:2]

    scale = min(max_height / dh, 1.0)
    if scale < 1.0:
        drawing_resized = cv2.resize(drawing_with_boxes, (int(dw * scale), int(dh * scale)))
    else:
        drawing_resized = drawing_with_boxes

    rh = drawing_resized.shape[0]
    p_scale = rh / ph
    pattern_resized = cv2.resize(pattern_bgr, (int(pw * p_scale), rh))

    combined = np.hstack([pattern_resized, drawing_resized])
    return combined


def format_results_json(detections):
    """Format detection results as a clean JSON-serializable list."""
    results = []
    for i, det in enumerate(detections):
        entry = {
            "id": i + 1,
            "bbox": {
                "x": det["x"],
                "y": det["y"],
                "width": det["w"],
                "height": det["h"],
            },
            "confidence": round(det["confidence"], 4),
        }
        if "scale" in det:
            entry["scale"] = round(det["scale"], 2)
        if "angle" in det:
            entry["angle"] = det["angle"]
        results.append(entry)
    return results
