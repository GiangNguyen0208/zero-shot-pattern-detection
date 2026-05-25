"""Gradio demo for zero-shot pattern detection in technical drawings."""

import json
import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image

from src.detector import PatternDetector

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV BGR format."""
    if pil_img is None:
        return None
    rgb = np.array(pil_img)
    if len(rgb.shape) == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    """Convert OpenCV BGR image to PIL Image."""
    if len(cv2_img.shape) == 2:
        return Image.fromarray(cv2_img)
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def detect_pattern(
    pattern_image,
    drawing_image,
    threshold,
    enable_rotation,
    enable_feature_matching,
    rotation_step,
):
    """Main detection function for Gradio interface."""
    if pattern_image is None or drawing_image is None:
        return None, "Please upload both a pattern image and a drawing image."

    pattern_cv = pil_to_cv2(pattern_image)
    drawing_cv = pil_to_cv2(drawing_image)

    if enable_rotation:
        step = int(rotation_step)
        angles = list(range(0, 360, step))
    else:
        angles = [0]

    detector = PatternDetector(
        threshold=threshold,
        angles=angles,
        use_feature_matching=enable_feature_matching,
    )

    result = detector.detect(drawing_cv, pattern_cv)

    vis_pil = cv2_to_pil(result["visualization"])

    results_text = json.dumps(
        {
            "num_detections": result["num_detections"],
            "inference_time_seconds": round(result["inference_time"], 2),
            "detections": result["results_json"],
        },
        indent=2,
    )

    return vis_pil, results_text


def build_demo():
    """Build and return the Gradio interface."""
    with gr.Blocks(
        title="Zero-Shot Pattern Detection in Technical Drawings",
    ) as demo:
        gr.Markdown(
            """
            # Zero-Shot Pattern Detection in Technical Drawings

            Upload a **pattern image** (the symbol/component to find) and a **drawing image**
            (the technical drawing to search in). The system will locate all occurrences
            of the pattern in the drawing using multi-scale template matching and feature-based matching.

            **Features:**
            - Multi-scale detection (0.5x - 2.0x)
            - Optional rotation invariance
            - Confidence scoring per detection
            - Works with any pattern — no training needed
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                pattern_input = gr.Image(
                    label="Pattern Image (query)",
                    type="pil",
                    height=250,
                )
                drawing_input = gr.Image(
                    label="Drawing Image (search target)",
                    type="pil",
                    height=400,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    threshold_slider = gr.Slider(
                        minimum=0.3,
                        maximum=0.95,
                        value=0.55,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Lower = more detections (may include false positives). Higher = fewer but more precise.",
                    )
                    enable_rotation = gr.Checkbox(
                        label="Enable Rotation Detection",
                        value=False,
                        info="Search for rotated versions of the pattern (slower).",
                    )
                    rotation_step = gr.Slider(
                        minimum=15,
                        maximum=90,
                        value=90,
                        step=15,
                        label="Rotation Step (degrees)",
                        info="Angle increment for rotation search. Smaller = more thorough but slower.",
                        visible=True,
                    )
                    enable_feature = gr.Checkbox(
                        label="Enable Feature Matching (ORB)",
                        value=True,
                        info="Use ORB feature matching as supplementary method.",
                    )

                detect_btn = gr.Button("Detect Patterns", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Detection Results",
                    type="pil",
                    height=500,
                )
                output_json = gr.Textbox(
                    label="Detection Details (JSON)",
                    lines=12,
                    max_lines=25,
                )

        detect_btn.click(
            fn=detect_pattern,
            inputs=[
                pattern_input,
                drawing_input,
                threshold_slider,
                enable_rotation,
                enable_feature,
                rotation_step,
            ],
            outputs=[output_image, output_json],
        )

        gr.Markdown("### Examples (click to load)")
        gr.Examples(
            examples=[
                [
                    os.path.join(EXAMPLES_DIR, "patterns", "example1_pattern.png"),
                    os.path.join(EXAMPLES_DIR, "drawings", "example1_drawing.png"),
                    0.55, False, True, 90,
                ],
                [
                    os.path.join(EXAMPLES_DIR, "patterns", "example2_pattern.png"),
                    os.path.join(EXAMPLES_DIR, "drawings", "example2_drawing.png"),
                    0.55, False, True, 90,
                ],
                [
                    os.path.join(EXAMPLES_DIR, "patterns", "example3_pattern.png"),
                    os.path.join(EXAMPLES_DIR, "drawings", "example3_drawing.png"),
                    0.55, False, True, 90,
                ],
            ],
            inputs=[
                pattern_input,
                drawing_input,
                threshold_slider,
                enable_rotation,
                enable_feature,
                rotation_step,
            ],
            outputs=[output_image, output_json],
            fn=detect_pattern,
            cache_examples=False,
            label="Click an example to load it",
        )

        gr.Markdown(
            """
            ---
            ### How it works
            1. **Preprocessing**: Images are converted to grayscale and contrast-enhanced (CLAHE)
            2. **Multi-scale Template Matching**: The pattern is resized to multiple scales and matched using Normalized Cross-Correlation (NCC)
            3. **Feature Matching (optional)**: ORB keypoints are extracted and matched for rotation/scale robustness
            4. **Post-processing**: Non-Maximum Suppression removes duplicate detections; results are filtered by aspect ratio
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False, theme=gr.themes.Soft())
