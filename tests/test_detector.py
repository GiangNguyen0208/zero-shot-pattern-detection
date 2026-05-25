"""Tests for the pattern detection pipeline."""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import PatternDetector
from src.preprocessing import preprocess_image, preprocess_pattern, to_grayscale
from src.template_matching import multi_scale_template_match, rotate_image
from src.postprocessing import non_max_suppression, compute_iou


def test_preprocessing():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(img)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    processed = preprocess_image(img)
    assert len(processed.shape) == 2
    print("[PASS] test_preprocessing")


def test_rotation():
    img = np.ones((50, 80), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 10), (70, 40), 0, 2)
    rotated = rotate_image(img, 45)
    assert rotated.shape[0] > 0 and rotated.shape[1] > 0
    print("[PASS] test_rotation")


def test_nms():
    dets = [
        {"x": 10, "y": 10, "w": 50, "h": 50, "confidence": 0.9},
        {"x": 15, "y": 15, "w": 50, "h": 50, "confidence": 0.8},
        {"x": 200, "y": 200, "w": 50, "h": 50, "confidence": 0.7},
    ]
    result = non_max_suppression(dets, iou_threshold=0.3)
    assert len(result) == 2
    assert result[0]["confidence"] == 0.9
    assert result[1]["confidence"] == 0.7
    print("[PASS] test_nms")


def test_iou():
    box1 = {"x": 0, "y": 0, "w": 10, "h": 10}
    box2 = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert abs(compute_iou(box1, box2) - 1.0) < 1e-6

    box3 = {"x": 100, "y": 100, "w": 10, "h": 10}
    assert compute_iou(box1, box3) == 0.0
    print("[PASS] test_iou")


def test_detection_on_synthetic():
    drawing = np.ones((400, 600), dtype=np.uint8) * 255
    pattern = np.ones((40, 40), dtype=np.uint8) * 255

    cv2.rectangle(pattern, (5, 5), (35, 35), 0, 2)
    cv2.line(pattern, (5, 5), (35, 35), 0, 2)
    cv2.line(pattern, (35, 5), (5, 35), 0, 2)

    positions = [(50, 50), (200, 100), (400, 250)]
    for px, py in positions:
        cv2.rectangle(drawing, (px + 5, py + 5), (px + 35, py + 35), 0, 2)
        cv2.line(drawing, (px + 5, py + 5), (px + 35, py + 35), 0, 2)
        cv2.line(drawing, (px + 35, py + 5), (px + 5, py + 35), 0, 2)

    detector = PatternDetector(
        threshold=0.6,
        use_feature_matching=False,
    )
    result = detector.detect(drawing, pattern)

    print(f"  Found {result['num_detections']} detections (expected ~{len(positions)})")
    print(f"  Inference time: {result['inference_time']:.3f}s")

    assert result["num_detections"] >= 2, f"Expected at least 2, got {result['num_detections']}"
    print("[PASS] test_detection_on_synthetic")


def test_detection_on_examples():
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

    for i in range(1, 4):
        drawing_path = os.path.join(examples_dir, "drawings", f"example{i}_drawing.png")
        pattern_path = os.path.join(examples_dir, "patterns", f"example{i}_pattern.png")

        if not os.path.exists(drawing_path) or not os.path.exists(pattern_path):
            print(f"  [SKIP] Example {i} not found")
            continue

        detector = PatternDetector(threshold=0.55, use_feature_matching=True)
        result = detector.detect(drawing_path, pattern_path)

        print(f"  Example {i}: {result['num_detections']} detections in {result['inference_time']:.2f}s")

        vis_path = os.path.join(examples_dir, f"result_example{i}.png")
        cv2.imwrite(vis_path, result["visualization"])

    print("[PASS] test_detection_on_examples")


if __name__ == "__main__":
    print("Running tests...\n")
    test_preprocessing()
    test_rotation()
    test_nms()
    test_iou()
    test_detection_on_synthetic()
    test_detection_on_examples()
    print("\nAll tests passed!")
