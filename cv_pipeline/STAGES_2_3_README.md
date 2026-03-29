# Stage 2 & 3: Face Detection and Segmentation

This directory contains the implementation for Stage 2 (Face Detection) and Stage 3 (Face Segmentation) of the BJJU CV pipeline.

## Stage 2: Face Detection (YOLOv8-face)

### Overview
Detects faces in group photos using YOLOv8-face with automatic inference strategy selection based on image size.

### Features
- **Adaptive Inference**: Automatically selects direct, scaled, or tiled inference based on image dimensions
- **VRAM Optimization**: FP16 inference (~0.8GB VRAM) with explicit model unloading
- **Multi-scale Support**: Handles images from 640px to 4000px+
- **Quality Filtering**: Filters detections by confidence, size, and aspect ratio
- **BBox Expansion**: Automatically expands bboxes for SAM to capture hair region

### Key Functions

```python
from cv_pipeline.stage2_detection.detector import (
    load_detection_model,
    unload_detection_model,
    process_image
)

# Load model
model = load_detection_model(config)

# Process image
result = process_image(image_path, image_metadata, model, config)

# Unload before Stage 3
unload_detection_model(model)
```

### Configuration

```yaml
detection:
  model:
    weights_path: "/app/models/yolov8/yolov8n-face.pt"
    device: "cuda:0"
    half: true  # FP16 mode
  
  inference:
    imgsz: 640
    conf: 0.5
    iou: 0.45
    max_det: 100
  
  filtering:
    min_confidence: 0.5
    min_face_size: 30
    max_face_size: 1000
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
```

### Output

Each image produces a `DetectionResult` containing:
- Accepted detections with expanded bboxes
- Rejected detections with reasons
- Debug visualization (optional)
- Processing metrics

---

## Stage 3: Face Segmentation (MobileSAM)

### Overview
Segments individual faces using MobileSAM with bounding box prompts from Stage 2, producing high-quality isolated face images on black backgrounds.

### Features
- **Efficient Segmentation**: MobileSAM in FP16 (~1.5GB VRAM)
- **Batch Processing**: Processes multiple faces efficiently by computing image embedding once
- **Mask Refinement**: Morphological operations for clean edges
- **Quality Control**: Minimum 112x112 dimensions for AdaFace compatibility
- **Metadata Output**: Comprehensive JSON metadata for each face

### Key Functions

```python
from cv_pipeline.stage3_segmentation.segmenter import (
    load_segmentation_model,
    unload_segmentation_model,
    process_image
)

# Load model (after YOLO unloaded)
predictor = load_segmentation_model(config)

# Process image
session_result = process_image(
    image_path,
    det_result,
    predictor,
    config,
    output_dir
)

# Unload when done
unload_segmentation_model(predictor)
```

### Configuration

```yaml
segmentation:
  model:
    type: "mobilesam"
    checkpoint_path: "/app/models/mobilesam/mobile_sam.pt"
    device: "cuda:0"
    half: true  # FP16 mode
  
  inference:
    multimask_output: true
    batch_size: 8
  
  refinement:
    enabled: true
    open_kernel_size: 3
    close_kernel_size: 5
    fill_holes: true
    keep_largest: true
    smooth_edges: true
  
  output:
    format: "png"
    min_dimension: 112
    background_color: [0, 0, 0]  # Black
```

### Output Structure

```
./output/
├── masks/
│   └── 20231027/                    # Date (YYYYMMDD)
│       └── session_1/
│           ├── 202310271_face_001_a1b2c3.png
│           ├── 202310271_face_001_a1b2c3.json
│           ├── 202310271_face_002_d4e5f6.png
│           └── ...
├── metadata/
│   └── 20231027_session_1_manifest.json
└── debug/
    ├── detections/
    │   └── 202310271_detections.jpg
    └── segmentations/
        └── 202310271_segmentation_overlay.jpg
```

### Metadata Schema

Each face has a JSON sidecar file:

```json
{
  "version": "1.0",
  "generated_at": "2026-03-28T14:30:00.000Z",
  "source": {
    "filename": "202310271.jpg",
    "date": "2023-10-27",
    "session": 1
  },
  "detection": {
    "detection_id": "202310271_001",
    "confidence": 0.92
  },
  "segmentation": {
    "sam_score": 0.95,
    "mask_area_pixels": 8432
  },
  "output": {
    "final_dimensions": [126, 158],
    "was_upscaled": false
  },
  "quality_flags": {
    "segmentation_quality": "good"
  }
}
```

---

## Hardware Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or equivalent
- **CUDA**: 11.7+ or 12.1+
- **RAM**: 16GB+ recommended

## VRAM Management

The pipeline uses **sequential model loading** to stay within 8GB VRAM:

1. **Stage 2 (Detection)**: Load YOLO → Process → **Unload** → Free ~0.8GB
2. **Stage 3 (Segmentation)**: Load SAM → Process → Unload → Free ~1.5GB

**Critical**: Always call `unload_detection_model()` before loading SAM!

## Performance

Typical processing times on RTX 4060:

| Image Size | Stage 2 (Detection) | Stage 3 (Segmentation) | Total |
|------------|---------------------|------------------------|-------|
| 1600x1200  | ~15ms              | ~45s (24 faces)        | ~45s  |
| 4032x3024  | ~120ms (tiled)     | ~60s (30 faces)        | ~60s  |

## Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics (YOLOv8)
pip install ultralytics

# Install MobileSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Install other dependencies
pip install -r requirements.txt
```

## Model Downloads

### YOLOv8-face
```bash
mkdir -p /app/models/yolov8
# Download from: https://github.com/derronqi/yolov8-face
# Place yolov8n-face.pt in /app/models/yolov8/
```

### MobileSAM
```bash
mkdir -p /app/models/mobilesam
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
  -O /app/models/mobilesam/mobile_sam.pt
```

## Usage Example

See `examples/run_detection_segmentation.py` for a complete pipeline example.

## Error Handling

Both stages implement comprehensive error handling:

- **CUDA OOM**: Automatic retry with reduced batch size or scaled inference
- **Empty Masks**: Skip and log, continue with other faces
- **File I/O Errors**: Graceful degradation with detailed logging

## Logging

Both stages use `loguru` for structured logging:

```
INFO  | Loading YOLOv8-face model | weights=yolov8n-face.pt | device=cuda:0
INFO  | YOLO model loaded | VRAM: 0.82 GB
INFO  | Detection complete | faces=24 | rejected=3 | time=12.5ms
INFO  | YOLO model unloaded | Remaining VRAM: 0.15 GB
INFO  | Loading MobileSAM model | checkpoint=mobile_sam.pt
INFO  | MobileSAM loaded | VRAM: 1.52 GB
INFO  | Segmented 22/24 faces successfully
INFO  | Session complete | time=45.2s
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in segmentation config
- Enable `aggressive_cleanup` in segmentation config
- Ensure YOLO is properly unloaded before SAM

### "No faces detected"
- Check image quality and resolution
- Lower `conf` threshold in detection config
- Verify model weights are correct

### "Empty mask returned"
- Face may be too small or occluded
- Check expanded bbox includes full head
- Review debug visualizations

## Next Steps

After Stage 3, face images are ready for:
- **Stage 4**: Face Recognition (AdaFace)
- **Stage 5**: HITL Review Interface
- **Stage 6**: Database Storage

---

For detailed architectural documentation, see:
- `docs/pipeline/02_Detection_Stage_Plan.md`
- `docs/pipeline/03_Segmentation_Stage_Plan.md`
