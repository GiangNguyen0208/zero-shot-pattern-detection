"""Multi-scale, multi-rotation template matching using OpenCV."""

import cv2
import numpy as np


def rotate_image(img, angle):
    """Rotate image by given angle (degrees), expanding canvas to fit."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)

    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2

    border_val = 255 if img.mean() > 127 else 0
    rotated = cv2.warpAffine(
        img, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


def _crop_to_content(img, padding=2):
    """Crop rotated template to its content bounding box."""
    bg_val = 255 if img.mean() > 127 else 0
    if bg_val == 255:
        mask = img < 240
    else:
        mask = img > 15

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(img.shape[0], y1 + padding + 1)
    x1 = min(img.shape[1], x1 + padding + 1)

    return img[y0:y1, x0:x1]


def match_at_single_scale(drawing, template, method=cv2.TM_CCOEFF_NORMED):
    """Run template matching at a single scale, return correlation map."""
    th, tw = template.shape[:2]
    dh, dw = drawing.shape[:2]

    if tw > dw or th > dh:
        return None, (tw, th)

    result = cv2.matchTemplate(drawing, template, method)
    return result, (tw, th)


def find_peaks(corr_map, threshold=0.5, template_size=(0, 0)):
    """Find all peaks above threshold in correlation map."""
    if corr_map is None:
        return []

    tw, th = template_size
    locations = np.where(corr_map >= threshold)
    detections = []

    for y, x in zip(*locations):
        score = float(corr_map[y, x])
        detections.append({
            "x": int(x),
            "y": int(y),
            "w": int(tw),
            "h": int(th),
            "confidence": score,
        })

    return detections


def multi_scale_template_match(
    drawing,
    template,
    scales=None,
    angles=None,
    threshold=0.55,
    method=cv2.TM_CCOEFF_NORMED,
):
    """Multi-scale, multi-rotation template matching.

    Args:
        drawing: Preprocessed drawing image (grayscale).
        template: Preprocessed pattern image (grayscale).
        scales: List of scale factors to try. Default covers 0.5x-2.0x.
        angles: List of rotation angles (degrees). Default is [0].
        threshold: Minimum correlation score.
        method: OpenCV template matching method.

    Returns:
        List of detection dicts with x, y, w, h, confidence, scale, angle.
    """
    if scales is None:
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
    if angles is None:
        angles = [0]

    dh, dw = drawing.shape[:2]
    th, tw = template.shape[:2]
    all_detections = []

    for angle in angles:
        if angle != 0:
            rot_template = rotate_image(template, angle)
            rot_template = _crop_to_content(rot_template)
        else:
            rot_template = template

        for scale in scales:
            new_tw = max(1, int(rot_template.shape[1] * scale))
            new_th = max(1, int(rot_template.shape[0] * scale))

            if new_tw > dw or new_th > dh:
                continue
            if new_tw < 10 or new_th < 10:
                continue

            scaled = cv2.resize(
                rot_template, (new_tw, new_th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )

            corr_map, tsize = match_at_single_scale(drawing, scaled, method)
            if corr_map is None:
                continue

            detections = find_peaks(corr_map, threshold, tsize)
            for d in detections:
                d["scale"] = scale
                d["angle"] = angle
            all_detections.extend(detections)

    return all_detections
