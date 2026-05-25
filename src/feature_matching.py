"""Feature-based matching using ORB descriptors for rotation/scale-robust detection."""

import cv2
import numpy as np


def extract_orb_features(img, max_features=1000):
    """Extract ORB keypoints and descriptors."""
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio_threshold=0.75):
    """Match descriptors using BFMatcher with ratio test."""
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)
    return good


def find_homography_regions(kp_pattern, kp_drawing, good_matches, pattern_shape, min_matches=6):
    """Find pattern location(s) using homography estimation.

    Uses RANSAC to robustly estimate the transformation from pattern to drawing.
    """
    if len(good_matches) < min_matches:
        return []

    src_pts = np.float32([kp_pattern[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_drawing[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return []

    inlier_count = int(mask.sum())
    confidence = min(1.0, inlier_count / max(min_matches, 1))

    ph, pw = pattern_shape[:2]
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)
    transformed = transformed.reshape(-1, 2)

    x_min = max(0, int(transformed[:, 0].min()))
    y_min = max(0, int(transformed[:, 1].min()))
    x_max = int(transformed[:, 0].max())
    y_max = int(transformed[:, 1].max())

    w = x_max - x_min
    h = y_max - y_min

    if w < 5 or h < 5:
        return []

    return [{
        "x": x_min,
        "y": y_min,
        "w": w,
        "h": h,
        "confidence": confidence,
        "method": "feature_matching",
        "inliers": inlier_count,
    }]


def feature_based_match(drawing, pattern, max_features=2000, ratio_threshold=0.75, min_matches=6):
    """Full feature-based matching pipeline.

    Detects the pattern in the drawing using ORB features + homography.
    This method is more robust to rotation and scale changes but may find
    fewer instances than template matching.
    """
    kp_pattern, desc_pattern = extract_orb_features(pattern, max_features)
    kp_drawing, desc_drawing = extract_orb_features(drawing, max_features)

    good_matches = match_features(desc_pattern, desc_drawing, ratio_threshold)

    if len(good_matches) < min_matches:
        return []

    detections = find_homography_regions(
        kp_pattern, kp_drawing, good_matches, pattern.shape, min_matches
    )

    return detections


def sliding_window_feature_match(
    drawing, pattern, window_scale=2.0, stride_ratio=0.5,
    max_features=500, ratio_threshold=0.75, min_matches=4,
):
    """Sliding window + feature matching for finding multiple instances.

    Divides the drawing into overlapping windows and runs feature matching
    on each window. Better for finding multiple instances of the same pattern.
    """
    ph, pw = pattern.shape[:2]
    dh, dw = drawing.shape[:2]

    win_h = int(ph * window_scale)
    win_w = int(pw * window_scale)
    stride_y = max(1, int(ph * stride_ratio))
    stride_x = max(1, int(pw * stride_ratio))

    all_detections = []

    for y in range(0, dh - win_h + 1, stride_y):
        for x in range(0, dw - win_w + 1, stride_x):
            window = drawing[y:y + win_h, x:x + win_w]
            dets = feature_based_match(window, pattern, max_features, ratio_threshold, min_matches)

            for d in dets:
                d["x"] += x
                d["y"] += y
                all_detections.append(d)

    return all_detections
