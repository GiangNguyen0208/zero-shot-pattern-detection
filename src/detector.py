"""Main pattern detector orchestrating the full detection pipeline."""

import time
import cv2
import numpy as np

from .preprocessing import load_image, preprocess_image, preprocess_pattern
from .template_matching import multi_scale_template_match
from .feature_matching import feature_based_match, sliding_window_feature_match
from .postprocessing import (
    non_max_suppression,
    merge_detections,
    filter_by_aspect_ratio,
    clip_to_image,
)
from .visualization import draw_detections, format_results_json


class PatternDetector:
    """Zero-shot pattern detector for technical drawings.

    Combines multi-scale template matching with feature-based matching
    for robust pattern detection in BOM-style technical drawings.
    """

    def __init__(
        self,
        scales=None,
        angles=None,
        threshold=0.55,
        iou_threshold=0.3,
        use_feature_matching=True,
        use_edges=False,
        denoise_strength=0,
    ):
        self.scales = scales or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
        self.angles = angles or [0]
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.use_feature_matching = use_feature_matching
        self.use_edges = use_edges
        self.denoise_strength = denoise_strength

    def detect(self, drawing_input, pattern_input):
        """Run full detection pipeline.

        Args:
            drawing_input: Drawing image (path, numpy array, or PIL Image).
            pattern_input: Pattern image (path, numpy array, or PIL Image).

        Returns:
            dict with keys:
                - detections: list of detection dicts
                - visualization: image with drawn boxes (BGR)
                - results_json: formatted results
                - inference_time: time in seconds
        """
        start_time = time.time()

        drawing_raw = load_image(drawing_input)
        pattern_raw = load_image(pattern_input)

        drawing_gray = preprocess_image(drawing_raw, use_edges=self.use_edges, denoise_strength=self.denoise_strength)
        pattern_gray = preprocess_pattern(pattern_raw, use_edges=self.use_edges)

        template_dets = multi_scale_template_match(
            drawing_gray,
            pattern_gray,
            scales=self.scales,
            angles=self.angles,
            threshold=self.threshold,
        )

        template_dets = non_max_suppression(template_dets, self.iou_threshold)

        feature_dets = []
        if self.use_feature_matching:
            drawing_for_feat = preprocess_image(drawing_raw, use_edges=False, denoise_strength=self.denoise_strength)
            pattern_for_feat = preprocess_pattern(pattern_raw, use_edges=False)
            feature_dets = feature_based_match(drawing_for_feat, pattern_for_feat)

        all_dets = merge_detections(template_dets, feature_dets, self.iou_threshold)
        all_dets = filter_by_aspect_ratio(all_dets, pattern_gray.shape, tolerance=0.6)
        all_dets = clip_to_image(all_dets, drawing_raw.shape)
        all_dets = non_max_suppression(all_dets, self.iou_threshold)

        all_dets.sort(key=lambda d: (d["y"], d["x"]))

        vis = draw_detections(drawing_raw, all_dets)

        inference_time = time.time() - start_time

        return {
            "detections": all_dets,
            "visualization": vis,
            "results_json": format_results_json(all_dets),
            "inference_time": inference_time,
            "num_detections": len(all_dets),
        }

    def detect_batch(self, drawing_input, pattern_inputs):
        """Detect multiple patterns on the same drawing.

        Args:
            drawing_input: Drawing image.
            pattern_inputs: List of pattern images.

        Returns:
            List of result dicts, one per pattern.
        """
        results = []
        for pattern in pattern_inputs:
            result = self.detect(drawing_input, pattern)
            results.append(result)
        return results
