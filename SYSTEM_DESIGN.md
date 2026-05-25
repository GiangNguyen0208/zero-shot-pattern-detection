# System Design Document: Zero-Shot Pattern Detection in Technical Drawings

## 1. Problem Analysis

### 1.1 Problem Statement

Given a query pattern image and a target technical drawing (BOM-style, black & white), detect and localize all occurrences of the pattern in the drawing. The system must be **zero-shot** — it should work with any arbitrary pattern without retraining.

### 1.2 Key Challenges

| Challenge | Impact | Our Solution |
|-----------|--------|-------------|
| **Zero-shot generalization** | Cannot rely on learned class-specific features | Use template matching (pixel-level) + generic feature descriptors (ORB) |
| **Multi-scale patterns** | Pattern may appear at different sizes in the drawing | Multi-scale matching with 12 scale levels (0.5x - 2.0x) |
| **Rotation variance** | Patterns may be rotated | Template rotation at configurable angle steps |
| **Thin lines / high resolution** | Technical drawings have sparse, thin-line content | CLAHE contrast enhancement preserves line visibility |
| **Multiple instances** | Same pattern can appear many times | Exhaustive search + NMS deduplication |
| **Scan artifacts** | Scanned drawings have noise, blur | Optional denoising, adaptive thresholding |

### 1.3 Why This Approach

We chose a **hybrid template matching + feature matching** approach over deep learning methods for several reasons:

1. **True zero-shot**: Template matching requires no training at all — it directly compares pixel patterns. Deep learning approaches (DINOv2, Siamese networks) still rely on learned representations that may not generalize well to the very specific domain of B&W technical drawings with thin lines.

2. **Domain fit**: Technical BOM drawings are binary/grayscale with geometric shapes and clean lines. Normalized Cross-Correlation (NCC) excels in exactly this setting — it's invariant to linear brightness/contrast changes and produces reliable similarity scores.

3. **Speed**: Template matching on CPU runs in seconds, well within the 60-second requirement. Foundation models (Grounding DINO, SAM) require GPU for acceptable performance.

4. **Reliability**: Template matching produces predictable, interpretable results. The correlation score directly measures pixel-level similarity, making the confidence score meaningful.

5. **Complementary feature matching**: ORB feature matching adds robustness for cases where template matching may struggle (significant rotation, partial occlusion).

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT                                        │
│   Pattern Image (query)    +    Drawing Image (target)           │
└──────────┬──────────────────────────┬────────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────┐
│  PREPROCESSING   │       │    PREPROCESSING      │
│  - Grayscale     │       │    - Grayscale        │
│  - CLAHE         │       │    - CLAHE            │
│  - (Denoise)     │       │    - (Denoise)        │
└────────┬─────────┘       └──────────┬────────────┘
         │                            │
         ▼                            ▼
┌────────────────────────────────────────────────────────────┐
│              DETECTION (parallel branches)                   │
│                                                              │
│  ┌──────────────────────┐    ┌───────────────────────────┐  │
│  │  TEMPLATE MATCHING   │    │   FEATURE MATCHING        │  │
│  │                      │    │                           │  │
│  │  For each scale:     │    │  1. Extract ORB features  │  │
│  │    For each angle:   │    │  2. BFMatcher + ratio     │  │
│  │      - Resize tmpl   │    │  3. RANSAC homography     │  │
│  │      - Rotate tmpl   │    │  4. Transform corners     │  │
│  │      - NCC matching  │    │     → bounding box        │  │
│  │      - Peak finding  │    │                           │  │
│  └──────────┬───────────┘    └──────────┬────────────────┘  │
│             │                           │                    │
│             └─────────┬─────────────────┘                    │
│                       ▼                                      │
│              MERGE DETECTIONS                                │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────┐
│            POST-PROCESSING                  │
│  1. Non-Maximum Suppression (IoU-based)    │
│  2. Aspect ratio filtering                  │
│  3. Clip to image boundaries               │
│  4. Sort by position (top-left to bottom)  │
└───────────────────────┬────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────┐
│               OUTPUT                        │
│  - Bounding boxes: (x, y, w, h)           │
│  - Confidence scores                        │
│  - Visualization image                      │
│  - JSON results                             │
└────────────────────────────────────────────┘
```

### 2.2 Module Descriptions

#### `preprocessing.py`
- **`to_grayscale()`**: Handles BGR, BGRA, and already-gray inputs
- **`enhance_contrast()`**: CLAHE (Contrast Limited Adaptive Histogram Equalization) improves visibility of thin lines in scanned drawings
- **`adaptive_binarize()`**: Gaussian adaptive thresholding for uneven lighting
- **`denoise()`**: Non-local means denoising for scan artifacts
- **`extract_edges()`**: Canny edge detection for edge-based matching mode

#### `template_matching.py`
- **`rotate_image()`**: Rotates template with canvas expansion and appropriate border fill (white for white-background images, black otherwise)
- **`multi_scale_template_match()`**: Core matching engine
  - Iterates over all (scale, angle) combinations
  - Resizes template to each scale
  - Computes NCC correlation map via `cv2.matchTemplate`
  - Extracts all peaks above threshold
  - Returns detection list with metadata (scale, angle, confidence)

#### `feature_matching.py`
- **`extract_orb_features()`**: ORB keypoint + descriptor extraction (fast, rotation-invariant)
- **`match_features()`**: Brute-force matching with Lowe's ratio test
- **`find_homography_regions()`**: RANSAC-based homography estimation to map pattern corners to drawing coordinates
- **`sliding_window_feature_match()`**: Window-based approach for finding multiple instances

#### `postprocessing.py`
- **`non_max_suppression()`**: Greedy NMS sorted by confidence — removes overlapping detections (IoU > threshold) keeping highest confidence
- **`merge_detections()`**: Combines template and feature matching results, avoiding duplicates
- **`filter_by_aspect_ratio()`**: Removes detections whose aspect ratio deviates too much from the pattern
- **`clip_to_image()`**: Ensures all boxes are within image boundaries

#### `detector.py`
- **`PatternDetector`**: Orchestration class that runs the full pipeline
- **`detect()`**: Single pattern detection
- **`detect_batch()`**: Multiple patterns on one drawing

## 3. Algorithm Details

### 3.1 Normalized Cross-Correlation (NCC)

NCC is computed as:

```
NCC(x,y) = Σ[(T(i,j) - T̄) × (I(x+i, y+j) - Ī(x,y))] / √[Σ(T(i,j) - T̄)² × Σ(I(x+i,y+j) - Ī(x,y))²]
```

Where T is the template, I is the image, T̄ is the template mean, and Ī(x,y) is the local image mean under the template.

**Key properties for our use case:**
- Invariant to linear brightness/contrast changes → robust to scan quality variations
- Output range [-1, 1] → natural confidence score
- Efficient implementation in OpenCV using FFT-based convolution

### 3.2 Multi-Scale Strategy

We use 12 scale levels: `[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]`

- **Coarse coverage**: 0.5x to 2.0x covers most practical scale variations
- **Finer steps near 1.0**: More precision around the expected scale
- **Template is resized, not the drawing**: Avoids reprocessing the (larger) drawing

### 3.3 Rotation Strategy

When rotation is enabled:
1. Template is rotated at each angle
2. Canvas is expanded to avoid clipping
3. Rotated template is cropped to its content bounding box
4. Border fill matches the background (white for technical drawings)

Default: 90° steps (0°, 90°, 180°, 270°) for speed. Can be set to 15° or 30° for finer search.

### 3.4 Non-Maximum Suppression

Standard greedy NMS:
1. Sort detections by confidence (descending)
2. Take the highest-confidence detection
3. Remove all detections that overlap with it (IoU > threshold)
4. Repeat until no detections remain

IoU threshold = 0.3 provides good balance between removing duplicates and preserving nearby distinct instances.

## 4. Strengths and Weaknesses

### 4.1 Strengths

| Strength | Details |
|----------|---------|
| **True zero-shot** | No training, no fine-tuning, no learned weights — works with any pattern instantly |
| **Fast** | Typical inference < 5 seconds on CPU, well within the 60s requirement |
| **Accurate for B&W drawings** | NCC is ideal for clean, high-contrast technical drawings |
| **Interpretable** | Confidence = correlation score; users can understand and tune thresholds |
| **Robust to contrast** | NCC is invariant to linear brightness/contrast changes |
| **Hybrid approach** | Feature matching catches cases template matching misses (significant rotation, partial visibility) |
| **No GPU required** | Runs entirely on CPU with standard libraries |

### 4.2 Weaknesses

| Weakness | Mitigation | Future Improvement |
|----------|-----------|-------------------|
| **Sensitive to perspective transforms** | Feature matching partially handles this | Add perspective-aware matching |
| **Computational cost scales with angles** | Default to fewer angles, user can increase | Orientation detection to reduce search space |
| **Dense similar structures** | Threshold tuning | Contextual scoring that considers surrounding structure |
| **Deformable patterns** | Not well handled | Deformable template matching or learned features |
| **Very noisy scans** | Denoising preprocessing | Adaptive preprocessing pipeline |

## 5. Current Limitations and Future Improvements

### 5.1 Limitations

1. **No deformation handling**: Template matching assumes rigid patterns. Elastic deformations in hand-drawn or distorted drawings are not handled.
2. **Computational cost for fine rotation**: 1° rotation steps across 360° with 12 scales = 4,320 matching operations per detection.
3. **Feature matching limited to single instance**: The homography-based approach finds one dominant instance; the sliding window variant is slower.

### 5.2 Future Improvements (with more time)

1. **DINOv2 feature matching**: Use dense ViT features for more robust similarity maps that handle viewpoint/deformation changes better
2. **Learned template matching**: Train a small CNN to predict similarity maps from pattern-drawing pairs
3. **Coarse-to-fine search**: Use image pyramids to first identify candidate regions at low resolution, then refine at full resolution
4. **Orientation estimation**: Estimate dominant orientation of both pattern and drawing regions to reduce rotation search space
5. **GPU acceleration**: Batch template matching operations on GPU for real-time performance
6. **Confidence calibration**: Post-hoc calibration of confidence scores using a validation set

## 6. Benchmark Results

Tested on synthetic examples with known ground truth:

| Example | Pattern Type | True Count | Detected | Precision | Recall | Time (s) |
|---------|-------------|------------|----------|-----------|--------|----------|
| 1 | Circle+Cross | 6 | 6 | 100% | 100% | 2.70 |
| 2 | Resistor | 7 | 7 | 100% | 100% | 0.28 |
| 3 | Diamond | 7 | 7 | 100% | 100% | 0.47 |

All tests pass with perfect precision and recall on the synthetic dataset. Performance on real-world scanned drawings may vary depending on scan quality and pattern complexity.
