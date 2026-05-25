"""Command-line inference script for pattern detection."""

import argparse
import json
import cv2
import sys

from src.detector import PatternDetector


def main():
    parser = argparse.ArgumentParser(description="Zero-shot pattern detection in technical drawings")
    parser.add_argument("--drawing", required=True, help="Path to the drawing image")
    parser.add_argument("--pattern", required=True, help="Path to the pattern image")
    parser.add_argument("--output", default="output.png", help="Path to save visualization")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--rotation", action="store_true", help="Enable rotation detection")
    parser.add_argument("--rotation-step", type=int, default=90, help="Rotation step in degrees")
    parser.add_argument("--no-feature", action="store_true", help="Disable feature matching")
    parser.add_argument("--json-output", default=None, help="Save results as JSON")
    args = parser.parse_args()

    angles = [0]
    if args.rotation:
        angles = list(range(0, 360, args.rotation_step))

    detector = PatternDetector(
        threshold=args.threshold,
        angles=angles,
        use_feature_matching=not args.no_feature,
    )

    print(f"Processing: pattern={args.pattern}, drawing={args.drawing}")
    result = detector.detect(args.drawing, args.pattern)

    print(f"Found {result['num_detections']} pattern(s) in {result['inference_time']:.2f}s")

    for det in result["results_json"]:
        bbox = det["bbox"]
        print(f"  #{det['id']}: x={bbox['x']}, y={bbox['y']}, "
              f"w={bbox['width']}, h={bbox['height']}, "
              f"confidence={det['confidence']:.4f}")

    cv2.imwrite(args.output, result["visualization"])
    print(f"Visualization saved to: {args.output}")

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(result["results_json"], f, indent=2)
        print(f"JSON results saved to: {args.json_output}")


if __name__ == "__main__":
    main()
