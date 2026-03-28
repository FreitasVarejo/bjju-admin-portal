# 02 - Detection Stage Plan (YOLOv8-face)

This document details the face detection stage using YOLOv8-face, including model configuration, inference strategy, VRAM management, and post-processing logic for dense group photos.

---

## 1. Stage Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: FACE DETECTION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  Load Model   │───▶│   Inference  │───▶│    NMS &    │───▶│   Extract     │  │
│  │  (YOLOv8-face)│    │  (640x640)   │    │   Filter    │    │   Bboxes      │  │
│  └───────────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│        │                    │                   │                   │           │
│        ▼                    ▼                   ▼                   ▼           │
│   ~1.5GB VRAM          Batch/Tiled         Conf > 0.5           List of        │
│   FP16 mode            processing          IoU < 0.45           Detection      │
│                                                                  objects       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Selection & Configuration

### 2.1 YOLOv8-face Model Variants

| Model | Parameters | VRAM (FP16) | Speed (RTX 4060) | mAP@0.5 | Recommendation |
|-------|------------|-------------|------------------|---------|----------------|
| **yolov8n-face** | 3.2M | ~0.8GB | ~8ms | 90.2% | **Primary choice** |
| yolov8s-face | 11.2M | ~1.2GB | ~12ms | 92.1% | Higher accuracy |
| yolov8m-face | 25.9M | ~1.8GB | ~20ms | 93.5% | Not recommended (VRAM) |

**Selection: yolov8n-face**

**Rationale**:
- Smallest VRAM footprint (~0.8GB in FP16)
- Fastest inference time
- Sufficient accuracy for bounding box generation (prompts for SAM)
- Leaves maximum headroom for SAM model

### 2.2 Model Source Options

**Option A: yolov8-face Community Weights** (Recommended)
- Source: https://github.com/derronqi/yolov8-face
- Pre-trained on WiderFace dataset
- Includes facial landmark detection (optional)

**Option B: Ultralytics Base + Fine-tune**
- Source: Ultralytics hub
- Would require additional fine-tuning
- Not recommended for prototype phase

### 2.3 Model Loading Configuration

```
MODEL_CONFIG = {
    "weights_path": "/app/models/yolov8/yolov8n-face.pt",
    "device": "cuda:0",
    "half": True,                    # FP16 inference
    "verbose": False,
    
    # Inference settings
    "imgsz": 640,                    # Input size
    "conf": 0.5,                     # Confidence threshold
    "iou": 0.45,                     # NMS IoU threshold
    "max_det": 100,                  # Max detections per image
    "classes": [0],                  # Face class only
    
    # Memory optimization
    "agnostic_nms": False,
    "retina_masks": False,
}
```

---

## 3. VRAM Management Strategy

### 3.1 Memory Budget

```
┌─────────────────────────────────────────────────────────────────┐
│              DETECTION STAGE VRAM ALLOCATION                    │
│                    (8GB Total Budget)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Component                    │  Allocation   │  Notes          │
│  ────────────────────────────│───────────────│─────────────────│
│  YOLOv8n-face (FP16)         │  ~0.8 GB      │  Static         │
│  Input tensor (640x640x3)    │  ~0.01 GB     │  Per inference  │
│  Feature maps (intermediate) │  ~0.5 GB      │  Peak during fwd│
│  Output tensors              │  ~0.1 GB      │  Detections     │
│  PyTorch workspace           │  ~0.5 GB      │  CUDA overhead  │
│  ────────────────────────────│───────────────│─────────────────│
│  TOTAL PEAK                  │  ~2.0 GB      │                 │
│  AVAILABLE FOR SAM           │  ~6.0 GB      │  After unload   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Loading Strategy

```
FUNCTION load_detection_model(config: ModelConfig) -> YOLO:
    """
    Load YOLOv8-face model with memory optimization.
    """
    # Clear any existing GPU memory
    IF torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Load model
    model = YOLO(config.weights_path)
    
    # Move to GPU with FP16
    model.to(config.device)
    IF config.half:
        model.model.half()
    
    # Warm-up inference (allocates CUDA memory)
    dummy_input = torch.zeros(1, 3, 640, 640).to(config.device)
    IF config.half:
        dummy_input = dummy_input.half()
    _ = model.predict(dummy_input, verbose=False)
    
    # Log memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    LOG_INFO(f"YOLO model loaded | VRAM: {allocated:.2f} GB")
    
    RETURN model
```

### 3.3 Model Unloading Strategy

```
FUNCTION unload_detection_model(model: YOLO) -> None:
    """
    Explicitly unload YOLO model and free GPU memory.
    Critical for sequential model loading strategy.
    """
    # Delete model reference
    del model
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Verify memory freed
    allocated = torch.cuda.memory_allocated() / 1024**3
    LOG_INFO(f"YOLO model unloaded | Remaining VRAM: {allocated:.2f} GB")
```

---

## 4. Inference Strategy

### 4.1 Standard Inference (Images ≤ 2048px)

```
FUNCTION detect_faces(
    image: np.ndarray,
    model: YOLO,
    config: DetectionConfig
) -> List[Detection]:
    """
    Run face detection on a single image.
    """
    # Run inference
    results = model.predict(
        source=image,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        max_det=config.max_det,
        half=config.half,
        verbose=False,
        device=config.device
    )
    
    # Extract detections
    detections = []
    FOR result IN results:
        boxes = result.boxes
        FOR i IN range(len(boxes)):
            detection = Detection(
                bbox=boxes.xyxy[i].cpu().numpy(),      # [x1, y1, x2, y2]
                confidence=boxes.conf[i].cpu().item(),
                class_id=int(boxes.cls[i].cpu().item()),
                # Optional: landmarks if available
                landmarks=extract_landmarks(result, i) IF has_landmarks ELSE None
            )
            detections.append(detection)
    
    LOG_DEBUG(f"Detected {len(detections)} faces")
    RETURN detections
```

### 4.2 Tiled Inference (High-Resolution Images)

For images larger than the model's native resolution, use Slicing-Aided Hyper Inference (SAHI) or manual tiling:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       TILED INFERENCE STRATEGY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    Original Image (4032 x 3024)                                                 │
│    ┌─────────────────────────────────────────────────────────────────┐          │
│    │                                                                  │          │
│    │   ┌────────────┬────────────┬────────────┬────────────┐        │          │
│    │   │   Tile 1   │   Tile 2   │   Tile 3   │   Tile 4   │        │          │
│    │   │  (640x640) │  (640x640) │  (640x640) │  (640x640) │        │          │
│    │   │   overlap  │   overlap  │   overlap  │   overlap  │        │          │
│    │   ├────────────┼────────────┼────────────┼────────────┤        │          │
│    │   │   Tile 5   │   Tile 6   │   Tile 7   │   Tile 8   │        │          │
│    │   │            │            │            │            │        │          │
│    │   └────────────┴────────────┴────────────┴────────────┘        │          │
│    │                                                                  │          │
│    └─────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
│    Process: Run inference on each tile, merge detections, apply global NMS     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Tiling Parameters**:
- Tile size: 640x640 (model native)
- Overlap: 128px (20%) to catch faces at tile boundaries
- Scale factor: Process at 1x, optionally add 0.5x for small faces

```
FUNCTION detect_faces_tiled(
    image: np.ndarray,
    model: YOLO,
    config: DetectionConfig,
    tile_size: int = 640,
    overlap: int = 128
) -> List[Detection]:
    """
    Tiled inference for high-resolution images.
    Prevents memory issues and improves small face detection.
    """
    height, width = image.shape[:2]
    all_detections = []
    
    # Calculate tile grid
    stride = tile_size - overlap
    n_tiles_x = math.ceil((width - overlap) / stride)
    n_tiles_y = math.ceil((height - overlap) / stride)
    
    LOG_DEBUG(f"Processing {n_tiles_x * n_tiles_y} tiles ({n_tiles_x}x{n_tiles_y})")
    
    FOR row IN range(n_tiles_y):
        FOR col IN range(n_tiles_x):
            # Calculate tile coordinates
            x1 = col * stride
            y1 = row * stride
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Pad if necessary (edge tiles)
            IF tile.shape[0] < tile_size OR tile.shape[1] < tile_size:
                tile = pad_to_size(tile, tile_size)
            
            # Run detection on tile
            tile_detections = detect_faces(tile, model, config)
            
            # Transform coordinates to original image space
            FOR det IN tile_detections:
                det.bbox[0] += x1  # x1
                det.bbox[1] += y1  # y1
                det.bbox[2] += x1  # x2
                det.bbox[3] += y1  # y2
                all_detections.append(det)
    
    # Merge overlapping detections across tiles
    merged_detections = merge_tile_detections(all_detections, iou_threshold=0.5)
    
    LOG_DEBUG(f"Tiled detection: {len(all_detections)} raw -> {len(merged_detections)} merged")
    
    RETURN merged_detections
```

### 4.3 Inference Decision Logic

```
FUNCTION detect_faces_auto(
    image: np.ndarray,
    model: YOLO,
    config: DetectionConfig
) -> List[Detection]:
    """
    Automatically choose inference strategy based on image size.
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)
    
    # Decision thresholds
    DIRECT_INFERENCE_MAX = 1280      # Use direct inference
    SCALED_INFERENCE_MAX = 2048      # Scale down then infer
    TILED_INFERENCE_MIN = 2048       # Use tiling
    
    IF max_dim <= DIRECT_INFERENCE_MAX:
        LOG_DEBUG("Using direct inference")
        RETURN detect_faces(image, model, config)
    
    ELIF max_dim <= SCALED_INFERENCE_MAX:
        LOG_DEBUG("Using scaled inference")
        scale = DIRECT_INFERENCE_MAX / max_dim
        scaled = cv2.resize(image, None, fx=scale, fy=scale)
        detections = detect_faces(scaled, model, config)
        # Scale bboxes back
        FOR det IN detections:
            det.bbox /= scale
        RETURN detections
    
    ELSE:
        LOG_DEBUG("Using tiled inference")
        RETURN detect_faces_tiled(image, model, config)
```

---

## 5. Post-Processing & Filtering

### 5.1 Detection Data Structure

```
CLASS Detection:
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    confidence: float         # Detection confidence (0-1)
    class_id: int             # Always 0 for face
    landmarks: Optional[np.ndarray]  # 5-point facial landmarks (if available)
    
    # Computed properties
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)
```

### 5.2 Non-Maximum Suppression (NMS)

YOLOv8 applies NMS internally, but additional NMS may be needed after tiled inference:

```
FUNCTION apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.45
) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    """
    IF len(detections) == 0:
        RETURN []
    
    # Convert to format for NMS
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    
    # Apply NMS (using torchvision or OpenCV)
    keep_indices = nms(
        torch.tensor(boxes),
        torch.tensor(scores),
        iou_threshold
    ).numpy()
    
    RETURN [detections[i] for i in keep_indices]
```

### 5.3 False Positive Filtering

In dense Jiu-Jitsu group photos, false positives may include:
- Patches on gis that resemble faces
- Partially occluded faces that shouldn't be processed
- Very small/distant faces (unreliable for recognition)

**Filtering Criteria**:

```
FUNCTION filter_detections(
    detections: List[Detection],
    config: FilterConfig
) -> Tuple[List[Detection], List[Detection]]:
    """
    Filter detections based on quality criteria.
    Returns (accepted, rejected) detections.
    """
    accepted = []
    rejected = []
    
    FOR det IN detections:
        rejection_reason = None
        
        # Filter 1: Confidence threshold
        IF det.confidence < config.min_confidence:
            rejection_reason = f"low_confidence:{det.confidence:.2f}"
        
        # Filter 2: Minimum size (in pixels)
        ELIF det.width < config.min_face_width OR det.height < config.min_face_height:
            rejection_reason = f"too_small:{det.width:.0f}x{det.height:.0f}"
        
        # Filter 3: Aspect ratio sanity (faces are roughly 0.7-1.3 W:H)
        ELIF det.aspect_ratio < 0.5 OR det.aspect_ratio > 2.0:
            rejection_reason = f"bad_aspect:{det.aspect_ratio:.2f}"
        
        # Filter 4: Maximum size (probably not a face if huge)
        ELIF det.width > config.max_face_width OR det.height > config.max_face_height:
            rejection_reason = f"too_large:{det.width:.0f}x{det.height:.0f}"
        
        IF rejection_reason:
            det.rejection_reason = rejection_reason
            rejected.append(det)
            LOG_DEBUG(f"Rejected detection: {rejection_reason}")
        ELSE:
            accepted.append(det)
    
    LOG_INFO(f"Filtering: {len(accepted)} accepted, {len(rejected)} rejected")
    RETURN accepted, rejected
```

### 5.4 Filter Configuration

```yaml
# Detection filtering configuration
filter:
  min_confidence: 0.5           # Minimum detection confidence
  min_face_width: 30            # Minimum face width in pixels
  min_face_height: 30           # Minimum face height in pixels
  max_face_width: 1000          # Maximum face width (sanity check)
  max_face_height: 1000         # Maximum face height
  min_aspect_ratio: 0.5         # Minimum width/height ratio
  max_aspect_ratio: 2.0         # Maximum width/height ratio
```

---

## 6. Handling Dense Group Photos

### 6.1 Jiu-Jitsu Photo Characteristics

**Typical Scene**:
- 10-40 people in frame
- Multiple rows (kneeling in front, standing in back)
- Uniform appearance (white/blue gis)
- Variable face sizes (near vs far from camera)
- Potential occlusions (shoulders, heads)

### 6.2 Multi-Scale Detection Strategy

```
FUNCTION detect_faces_multiscale(
    image: np.ndarray,
    model: YOLO,
    config: DetectionConfig
) -> List[Detection]:
    """
    Run detection at multiple scales to catch both large and small faces.
    """
    all_detections = []
    scales = [1.0, 0.5]  # Original and half resolution
    
    FOR scale IN scales:
        IF scale != 1.0:
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
        ELSE:
            scaled_image = image
        
        detections = detect_faces_auto(scaled_image, model, config)
        
        # Scale bboxes back to original resolution
        FOR det IN detections:
            det.bbox /= scale
            det.scale_detected = scale
        
        all_detections.extend(detections)
    
    # Merge detections from different scales
    merged = apply_nms(all_detections, iou_threshold=0.5)
    
    RETURN merged
```

### 6.3 Occlusion Handling Strategy

**For this prototype phase**: Accept partial occlusions, flag them for future HITL review.

```
FUNCTION assess_occlusion(
    detection: Detection,
    all_detections: List[Detection]
) -> float:
    """
    Estimate occlusion level based on overlapping detections.
    Returns occlusion score (0 = no occlusion, 1 = fully occluded).
    """
    occlusion_score = 0.0
    
    FOR other IN all_detections:
        IF other == detection:
            CONTINUE
        
        # Calculate IoU
        iou = calculate_iou(detection.bbox, other.bbox)
        
        # Check if current detection is below/behind another
        IF iou > 0.1:
            IF detection.center[1] > other.center[1]:  # Below in image
                # Lower face might be occluded by upper body of front person
                occlusion_score = max(occlusion_score, iou)
    
    detection.occlusion_score = occlusion_score
    RETURN occlusion_score
```

---

## 7. Detection Stage Output

### 7.1 Output Data Structure

```
CLASS DetectionResult:
    image_metadata: ImageMetadata     # From ingestion stage
    detections: List[Detection]       # Accepted detections
    rejected: List[Detection]         # Filtered out detections
    inference_time_ms: float          # Processing time
    detection_count: int              # Number of faces found
    
    # For debugging
    debug_image: Optional[np.ndarray] # Image with drawn bboxes

CLASS Detection:
    id: str                           # Unique ID (image_hash + index)
    bbox: np.ndarray                  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[np.ndarray]   # 5-point landmarks
    occlusion_score: float            # 0-1 occlusion estimate
    scale_detected: float             # Scale at which detected
    
    # For downstream stages
    bbox_expanded: np.ndarray         # Expanded bbox for SAM (includes hair region)
```

### 7.2 Bounding Box Expansion for Segmentation

SAM needs a slightly larger bounding box to capture the full head including hair:

```
FUNCTION expand_bbox_for_segmentation(
    bbox: np.ndarray,
    expansion_ratio: float = 0.3,
    image_shape: Tuple[int, int] = None
) -> np.ndarray:
    """
    Expand bounding box to include hair and provide context for SAM.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Expand more on top (for hair) and sides
    expand_top = height * expansion_ratio * 1.5    # Extra expansion for hair
    expand_bottom = height * expansion_ratio * 0.5  # Less expansion below chin
    expand_horizontal = width * expansion_ratio
    
    new_bbox = np.array([
        x1 - expand_horizontal,
        y1 - expand_top,
        x2 + expand_horizontal,
        y2 + expand_bottom
    ])
    
    # Clip to image boundaries
    IF image_shape:
        img_h, img_w = image_shape[:2]
        new_bbox = np.array([
            max(0, new_bbox[0]),
            max(0, new_bbox[1]),
            min(img_w, new_bbox[2]),
            min(img_h, new_bbox[3])
        ])
    
    RETURN new_bbox
```

---

## 8. Debug Visualization

### 8.1 Detection Visualization

```
FUNCTION draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    rejected: List[Detection] = None,
    show_landmarks: bool = True
) -> np.ndarray:
    """
    Draw detection bounding boxes on image for debugging.
    """
    vis_image = image.copy()
    
    # Draw accepted detections in green
    FOR i, det IN enumerate(detections):
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label with confidence
        label = f"{i+1}: {det.confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw landmarks if available
        IF show_landmarks AND det.landmarks IS NOT None:
            FOR point IN det.landmarks:
                cv2.circle(vis_image, tuple(point.astype(int)), 2, (255, 0, 0), -1)
    
    # Draw rejected detections in red
    IF rejected:
        FOR det IN rejected:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    RETURN vis_image
```

### 8.2 Saving Debug Output

```
FUNCTION save_detection_debug(
    result: DetectionResult,
    output_dir: Path,
    save_visualization: bool = True
) -> None:
    """
    Save detection results for debugging and verification.
    """
    # Create output directory
    debug_dir = output_dir / "debug" / "detections"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization image
    IF save_visualization:
        vis_path = debug_dir / f"{result.image_metadata.filename.stem}_detections.jpg"
        cv2.imwrite(str(vis_path), result.debug_image)
    
    # Save detection data as JSON
    json_path = debug_dir / f"{result.image_metadata.filename.stem}_detections.json"
    data = {
        "filename": result.image_metadata.filename,
        "detection_count": result.detection_count,
        "inference_time_ms": result.inference_time_ms,
        "detections": [
            {
                "id": det.id,
                "bbox": det.bbox.tolist(),
                "confidence": det.confidence,
                "occlusion_score": det.occlusion_score
            }
            FOR det IN result.detections
        ]
    }
    save_json(json_path, data)
```

---

## 9. Performance Optimization

### 9.1 Batch Processing (Optional)

For processing multiple images in a session, batch inference can improve throughput:

```
FUNCTION detect_faces_batch(
    images: List[np.ndarray],
    model: YOLO,
    config: DetectionConfig,
    batch_size: int = 4
) -> List[List[Detection]]:
    """
    Batch inference for multiple images.
    Note: Requires uniform image sizes or padding.
    """
    all_results = []
    
    FOR i IN range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Ensure uniform size (pad if necessary)
        batch_tensor = prepare_batch(batch, config.imgsz)
        
        # Run batch inference
        results = model.predict(
            batch_tensor,
            batch=True,
            half=config.half,
            verbose=False
        )
        
        all_results.extend(results)
    
    RETURN all_results
```

**Note**: For the prototype with sequential processing and memory constraints, single-image inference is recommended.

### 9.2 Warm-up and Caching

```
CLASS DetectionPipeline:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self._warmed_up = False
    
    def load(self):
        self.model = load_detection_model(self.config)
    
    def warmup(self):
        """Run warm-up inference to initialize CUDA kernels."""
        IF NOT self._warmed_up:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.detect(dummy)
            self._warmed_up = True
            LOG_INFO("Detection model warmed up")
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        ...
    
    def unload(self):
        unload_detection_model(self.model)
        self.model = None
        self._warmed_up = False
```

---

## 10. Error Handling

### 10.1 Detection Stage Errors

| Error | Cause | Recovery |
|-------|-------|----------|
| `CUDA OOM` | Image too large | Retry with tiling, reduce batch size |
| `No detections` | Empty image or all filtered | Log warning, return empty list |
| `Model load failure` | Corrupt weights | Re-download weights, fail gracefully |
| `Invalid image` | Corrupt input | Skip image, log error |

### 10.2 Graceful Degradation

```
FUNCTION safe_detect(
    image: np.ndarray,
    model: YOLO,
    config: DetectionConfig
) -> DetectionResult:
    """
    Run detection with error handling and fallbacks.
    """
    TRY:
        RETURN detect_faces_auto(image, model, config)
    
    EXCEPT torch.cuda.OutOfMemoryError:
        LOG_WARNING("OOM during detection, attempting recovery")
        torch.cuda.empty_cache()
        
        # Retry with smaller input
        scaled = cv2.resize(image, None, fx=0.5, fy=0.5)
        TRY:
            detections = detect_faces(scaled, model, config)
            FOR det IN detections:
                det.bbox *= 2  # Scale back
            RETURN detections
        EXCEPT:
            LOG_ERROR("Detection failed after OOM recovery")
            RETURN DetectionResult(detections=[], error="OOM")
    
    EXCEPT Exception AS e:
        LOG_ERROR(f"Detection failed: {e}")
        RETURN DetectionResult(detections=[], error=str(e))
```

---

## 11. Logging Specifications

```
# Detection stage log examples

INFO  | Loading YOLOv8-face model | weights=yolov8n-face.pt | device=cuda:0 | half=True
INFO  | Model loaded | VRAM=0.82GB | warmup_complete=True

DEBUG | Processing image | filename=202310271.jpg | resolution=1600x1200
DEBUG | Using direct inference | input_size=640x640
INFO  | Detection complete | faces=24 | rejected=3 | time=12.5ms

DEBUG | Rejected detection | reason=low_confidence:0.42 | bbox=[120,340,180,420]
DEBUG | Rejected detection | reason=too_small:25x28

WARN  | No faces detected | filename=202310272.jpg
WARN  | High rejection rate | accepted=5 | rejected=15 | ratio=0.25

INFO  | Unloading YOLO model | freed_vram=0.82GB
```

---

## 12. Configuration Reference

```yaml
# Detection stage configuration (subset of pipeline_config.yaml)

detection:
  model:
    type: "yolov8n-face"
    weights_path: "/app/models/yolov8/yolov8n-face.pt"
    device: "cuda:0"
    half: true                    # FP16 inference
  
  inference:
    imgsz: 640                    # Input size for YOLO
    conf: 0.5                     # Confidence threshold
    iou: 0.45                     # NMS IoU threshold
    max_det: 100                  # Max detections per image
    
  tiling:
    enabled: "auto"               # auto, always, never
    tile_size: 640
    overlap: 128
    
  multiscale:
    enabled: false                # Use multiscale detection
    scales: [1.0, 0.5]
  
  filtering:
    min_confidence: 0.5
    min_face_size: 30             # Minimum face dimension in pixels
    max_face_size: 1000
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.0
  
  output:
    bbox_expansion_ratio: 0.3     # Expansion for SAM prompts
    save_debug_visualizations: true
```

---

*Document Version: 1.0*  
*Last Updated: 2026-03-28*  
*Author: CV Pipeline Planning*
