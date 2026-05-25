# Zero-Shot Pattern Detection in Technical Drawings

A system for detecting arbitrary patterns in BOM-style technical drawings without any training or fine-tuning. Given a query pattern image and a target drawing, the system locates all occurrences of the pattern and returns bounding boxes with confidence scores.

## Features

- **Zero-shot detection**: Works with any pattern — no training required
- **Multi-scale matching**: Detects patterns at 0.5x to 2.0x scale
- **Rotation support**: Optional multi-angle search (configurable step)
- **Hybrid approach**: Combines template matching (NCC) with ORB feature matching
- **Fast inference**: < 5 seconds typical on CPU
- **Gradio demo**: Interactive web interface with visualization

## Project Structure

```
├── app.py                    # Gradio demo application
├── requirements.txt          # Python dependencies
├── generate_examples.py      # Script to generate example images
├── SYSTEM_DESIGN.md          # Detailed system design document
├── src/
│   ├── __init__.py
│   ├── detector.py           # Main PatternDetector class
│   ├── preprocessing.py      # Image preprocessing (grayscale, CLAHE, etc.)
│   ├── template_matching.py  # Multi-scale, multi-rotation template matching
│   ├── feature_matching.py   # ORB feature-based matching
│   ├── postprocessing.py     # NMS, bbox merging, filtering
│   └── visualization.py      # Drawing bboxes, formatting results
├── examples/
│   ├── patterns/             # Example pattern images
│   └── drawings/             # Example drawing images
└── tests/
    └── test_detector.py      # Unit and integration tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/GiangNguyen0208/zero-shot-pattern-detection.git
cd zero-shot-pattern-detection

# Create virtual environment (using uv)
uv venv .venv --python 3.11
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from src.detector import PatternDetector

detector = PatternDetector(
    threshold=0.55,       # Confidence threshold (0.3 - 0.95)
    angles=[0],           # Rotation angles to search
    use_feature_matching=True,
)

result = detector.detect("path/to/drawing.png", "path/to/pattern.png")

print(f"Found {result['num_detections']} matches")
print(f"Inference time: {result['inference_time']:.2f}s")

for det in result['results_json']:
    print(f"  #{det['id']}: bbox={det['bbox']}, confidence={det['confidence']}")

# result['visualization'] contains the image with drawn bounding boxes
import cv2
cv2.imwrite("output.png", result['visualization'])
```

### Gradio Demo

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Batch Detection

```python
patterns = ["pattern1.png", "pattern2.png", "pattern3.png"]
results = detector.detect_batch("drawing.png", patterns)
```

## Running Tests

```bash
python tests/test_detector.py
```

To regenerate example images:

```bash
python generate_examples.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.55 | Minimum confidence score (0.3 - 0.95) |
| `scales` | 0.5 - 2.0 | Scale factors for multi-scale matching |
| `angles` | [0] | Rotation angles in degrees |
| `use_feature_matching` | True | Enable ORB feature matching |
| `use_edges` | False | Use edge-based matching |
| `iou_threshold` | 0.3 | IoU threshold for NMS |
| `denoise_strength` | 0 | Denoising strength for scanned drawings |

## Approach

The system uses a **hybrid multi-scale template matching + feature matching** pipeline:

1. **Preprocessing**: Grayscale conversion, CLAHE contrast enhancement
2. **Multi-scale Template Matching**: NCC correlation at 12 scale levels (0.5x-2.0x)
3. **Feature Matching**: ORB descriptors + RANSAC homography estimation
4. **Post-processing**: NMS, aspect ratio filtering, bbox clipping

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for detailed analysis.

## Limitations

- Very small patterns (< 10x10 pixels) may not be reliably detected
- Heavily distorted or perspective-transformed patterns need the feature matching branch
- Dense drawings with many similar sub-structures may produce false positives at lower thresholds

## HuggingFace Demo

Live demo: [HuggingFace Space](https://huggingface.co/spaces/GiangNguyen0208/zero-shot-pattern-detection)

## License

MIT
