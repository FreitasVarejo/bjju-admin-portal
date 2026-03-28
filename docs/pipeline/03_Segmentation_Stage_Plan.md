# 03 - Segmentation Stage Plan (FastSAM/MobileSAM)

This document details the zero-shot segmentation stage using MobileSAM or FastSAM, including bounding box prompting, mask refinement for face+hair isolation, and the output I/O strategy for downstream HITL and AdaFace processing.

---

## 1. Stage Overview

```
+-----------------------------------------------------------------------------------+
|                        STAGE 3: SEGMENTATION                                       |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +---------------+    +----------------+    +--------------+    +---------------+ |
|  |  Load SAM     |--->|  Prompt with   |--->|  Generate    |--->|  Refine &     | |
|  |  (MobileSAM)  |    |  BBoxes        |    |  Raw Masks   |    |  Black BG     | |
|  +---------------+    +----------------+    +--------------+    +---------------+ |
|        |                    |                     |                   |           |
|        v                    v                     v                   v           |
|   ~1.5GB VRAM          Expanded            Per-face binary       Face+hair       |
|   FP16 mode            bboxes              segmentation          isolated        |
|                        from YOLO                                                  |
|                                                                                    |
|  +---------------+    +----------------+                                          |
|  |  Crop Face    |--->|  Save to Disk  |                                          |
|  |  Region       |    |  (PNG format)  |                                          |
|  +---------------+    +----------------+                                          |
|        |                    |                                                     |
|        v                    v                                                     |
|   Isolated face        ./output/masks/{date}/{session}/                           |
|   with black BG        {filename}_face_{idx}_{hash}.png                           |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Model Selection & Configuration

### 2.1 SAM Variant Comparison

| Model | Encoder | Parameters | VRAM (FP16) | Speed/Mask | Quality | Selection |
|-------|---------|------------|-------------|------------|---------|-----------|
| **MobileSAM** | TinyViT | 9.8M | ~1.5GB | ~50ms | Good | **PRIMARY** |
| FastSAM | YOLOv8-seg | 68M | ~2.5GB | ~80ms | Good | Fallback |
| SAM (ViT-B) | ViT-B | 93M | ~4GB | ~150ms | Excellent | Not viable |
| SAM (ViT-H) | ViT-H | 636M | ~7GB | ~500ms | Best | Not viable |

**Selection: MobileSAM**

**Rationale**:
- Lowest VRAM footprint among SAM variants (~1.5GB)
- Fastest inference time (~50ms per mask)
- Maintains good segmentation quality for face boundaries
- Leaves ~6GB VRAM headroom after YOLO unload
- Apache 2.0 license (commercial-friendly)

### 2.2 Model Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                          MobileSAM ARCHITECTURE                                    |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +-------------------+     +-----------------------+     +---------------------+   |
|  |   IMAGE ENCODER   |     |    PROMPT ENCODER     |     |   MASK DECODER      |   |
|  |     (TinyViT)     |     |                       |     |                     |   |
|  +-------------------+     +-----------------------+     +---------------------+   |
|          |                          |                            |                 |
|          v                          v                            v                 |
|    Image Embeddings          Prompt Embeddings             Output Masks            |
|    (256x64x64)               (Sparse + Dense)              (256x256 -> upscale)    |
|                                                                                    |
|  +-----------------------------------------------------------------------------------+
|  |                         PROMPTING MODES                                        |
|  |                                                                                 |
|  |  [x] Bounding Box  - Primary mode (from YOLOv8)                                |
|  |  [ ] Point prompts - Not used in this pipeline                                 |
|  |  [ ] Text prompts  - Not available in MobileSAM                                |
|  +-----------------------------------------------------------------------------------+
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

### 2.3 Model Loading Configuration

```
SAM_CONFIG = {
    "model_type": "mobilesam",
    "checkpoint_path": "/app/models/mobilesam/mobile_sam.pt",
    "device": "cuda:0",
    "half": True,                      # FP16 inference
    
    # Inference settings
    "points_per_side": None,           # Not using grid-based prompting
    "pred_iou_thresh": 0.88,           # IoU threshold for mask filtering
    "stability_score_thresh": 0.95,    # Stability threshold
    "crop_n_layers": 0,                # No multi-crop for box prompts
    
    # Output settings
    "multimask_output": True,          # Generate 3 masks, select best
    "return_logits": False,            # Return binary masks
}
```

---

## 3. VRAM Management Strategy

### 3.1 Memory Budget (After YOLO Unload)

```
+---------------------------------------------------------------------+
|              SEGMENTATION STAGE VRAM ALLOCATION                      |
|                   (8GB Total Budget)                                 |
+---------------------------------------------------------------------+
|                                                                      |
|  Pre-condition: YOLO model unloaded, ~0.2GB residual                |
|                                                                      |
|  Component                    |  Allocation   |  Notes              |
|  -----------------------------|---------------|---------------------|
|  MobileSAM Model (FP16)       |  ~1.5 GB      |  Static             |
|  Image Encoder Cache          |  ~0.3 GB      |  Per-image          |
|  Prompt Encoder               |  ~0.1 GB      |  Per-batch          |
|  Mask Decoder                 |  ~0.2 GB      |  Per-batch          |
|  Output masks (batch of 8)    |  ~0.3 GB      |  Peak               |
|  PyTorch workspace            |  ~0.5 GB      |  CUDA overhead      |
|  -----------------------------|---------------|---------------------|
|  TOTAL PEAK                   |  ~3.0 GB      |                     |
|  SAFETY BUFFER                |  ~5.0 GB      |  OOM prevention     |
|                                                                      |
+---------------------------------------------------------------------+
```

### 3.2 Model Loading Strategy

```
FUNCTION load_segmentation_model(config: SAMConfig) -> SamPredictor:
    """
    Load MobileSAM model after YOLO has been unloaded.
    """
    # Verify YOLO is unloaded
    current_vram = torch.cuda.memory_allocated() / 1024**3
    IF current_vram > 0.5:
        LOG_WARNING(f"High VRAM before SAM load: {current_vram:.2f}GB")
        torch.cuda.empty_cache()
    
    # Load MobileSAM
    sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint_path)
    
    # Move to GPU with FP16
    sam.to(config.device)
    IF config.half:
        sam = sam.half()
    
    # Create predictor wrapper
    predictor = SamPredictor(sam)
    
    # Warm-up inference
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    predictor.set_image(dummy_image)
    dummy_box = np.array([100, 100, 200, 200])
    _ = predictor.predict(box=dummy_box, multimask_output=True)
    
    # Log memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    LOG_INFO(f"MobileSAM loaded | VRAM: {allocated:.2f} GB")
    
    RETURN predictor
```

### 3.3 Sequential Model Loading Flow

```
+-----------------------------------------------------------------------------------+
|                     SEQUENTIAL MODEL LOADING STRATEGY                              |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|   Time --->                                                                        |
|                                                                                    |
|   VRAM    |                                                                        |
|   8GB ----|--------------------------------------------------------------------    |
|           |                                                                        |
|   6GB ----|--------------------------------------------------------------------    |
|           |                                                                        |
|   4GB ----|--------------------------------------------------------------------    |
|           |                                                                        |
|   2GB ----|----+----+              +----+                                          |
|           |    |YOLO|              |SAM |                                          |
|   0GB ----|----+----+--------------+----+----------------------------------        |
|           |                                                                        |
|           |<---DETECTION--->|<-SWAP->|<--------SEGMENTATION-------->|             |
|                             |        |                                             |
|                    empty_cache()     |                                             |
|                    gc.collect()      |                                             |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 4. Bounding Box Prompting Strategy

### 4.1 Prompt Preparation

The bounding boxes from YOLOv8 detection stage are used as prompts for SAM:

```
FUNCTION prepare_sam_prompts(
    detections: List[Detection],
    image_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Convert YOLO bounding boxes to SAM prompt format.
    Uses expanded bboxes to capture hair region.
    """
    prompts = []
    
    FOR det IN detections:
        # Use pre-expanded bbox from detection stage
        bbox = det.bbox_expanded
        
        # Ensure bbox is within image boundaries
        bbox = clip_bbox_to_image(bbox, image_shape)
        
        # SAM expects [x1, y1, x2, y2] format
        prompt = np.array([
            bbox[0],  # x1
            bbox[1],  # y1
            bbox[2],  # x2
            bbox[3]   # y2
        ])
        
        prompts.append({
            "box": prompt,
            "detection_id": det.id,
            "original_bbox": det.bbox,
            "confidence": det.confidence
        })
    
    RETURN prompts
```

### 4.2 Box Expansion Strategy for Hair Capture

```
+-----------------------------------------------------------------------------------+
|                        BOUNDING BOX EXPANSION                                      |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|    YOLO Detection (face only)          SAM Prompt (expanded for hair)             |
|                                                                                    |
|    +------------------+                 +------------------------+                 |
|    |                  |                 |      +30% TOP         |                 |
|    |                  |                 |      (hair region)     |                 |
|    |  +------------+  |                 |  +------------------+  |                 |
|    |  |            |  |                 |  |                  |  |                 |
|    |  |   FACE     |  |   -------->     |  |      FACE        |  |                 |
|    |  |   BBOX     |  |                 |  |      BBOX        |  |                 |
|    |  |            |  |                 |  |                  |  |                 |
|    |  +------------+  |                 |  +------------------+  |                 |
|    |                  |                 |     +15% BOTTOM       |                 |
|    +------------------+                 +------------------------+                 |
|                                         |<---+20% SIDES--->|                       |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

**Expansion Ratios**:
- Top: +45% of face height (for hair, especially long hair)
- Bottom: +15% of face height (minimal, just below chin)
- Sides: +30% of face width (for hair width)

```
FUNCTION expand_bbox_for_hair(
    bbox: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Expand face bbox to include hair region for SAM prompting.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Asymmetric expansion (more on top for hair)
    expand_top = height * 0.45
    expand_bottom = height * 0.15
    expand_horizontal = width * 0.30
    
    expanded = np.array([
        max(0, x1 - expand_horizontal),
        max(0, y1 - expand_top),
        min(image_shape[1], x2 + expand_horizontal),
        min(image_shape[0], y2 + expand_bottom)
    ])
    
    RETURN expanded
```

---

## 5. Mask Generation Pipeline

### 5.1 Single-Face Segmentation

```
FUNCTION segment_single_face(
    predictor: SamPredictor,
    image: np.ndarray,
    prompt: Dict,
    config: SegmentationConfig
) -> SegmentationResult:
    """
    Generate segmentation mask for a single face using box prompt.
    """
    # Set image (computes image embeddings)
    # Note: Only call once per image, reuse for multiple faces
    predictor.set_image(image)
    
    # Generate masks with box prompt
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt["box"],
        multimask_output=True  # Returns 3 masks
    )
    
    # Select best mask based on IoU score
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    LOG_DEBUG(f"Generated mask | score={best_score:.3f} | detection_id={prompt['detection_id']}")
    
    RETURN SegmentationResult(
        mask=best_mask,
        score=best_score,
        detection_id=prompt["detection_id"],
        bbox=prompt["box"],
        original_bbox=prompt["original_bbox"]
    )
```

### 5.2 Batch Processing Strategy

For multiple faces in a single image, process efficiently:

```
FUNCTION segment_all_faces(
    predictor: SamPredictor,
    image: np.ndarray,
    prompts: List[Dict],
    config: SegmentationConfig
) -> List[SegmentationResult]:
    """
    Segment all faces in an image efficiently.
    Image embedding is computed once and reused.
    """
    results = []
    
    # Compute image embedding ONCE
    predictor.set_image(image)
    image_embedding_computed = True
    
    # Process faces in batches for memory efficiency
    batch_size = config.batch_size  # Default: 8
    
    FOR i IN range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        FOR prompt IN batch_prompts:
            TRY:
                result = segment_single_face_no_embed(
                    predictor, prompt, config
                )
                results.append(result)
            EXCEPT Exception AS e:
                LOG_WARNING(f"Segmentation failed for {prompt['detection_id']}: {e}")
                results.append(SegmentationResult(
                    mask=None,
                    score=0.0,
                    detection_id=prompt["detection_id"],
                    error=str(e)
                ))
        
        # Optional: Clear intermediate tensors
        IF config.aggressive_memory_cleanup:
            torch.cuda.empty_cache()
    
    LOG_INFO(f"Segmented {len(results)} faces | success={sum(1 for r in results if r.mask is not None)}")
    
    RETURN results


FUNCTION segment_single_face_no_embed(
    predictor: SamPredictor,
    prompt: Dict,
    config: SegmentationConfig
) -> SegmentationResult:
    """
    Generate mask without recomputing image embedding.
    Assumes predictor.set_image() was already called.
    """
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt["box"],
        multimask_output=True
    )
    
    best_idx = np.argmax(scores)
    
    RETURN SegmentationResult(
        mask=masks[best_idx],
        score=scores[best_idx],
        detection_id=prompt["detection_id"],
        bbox=prompt["box"],
        original_bbox=prompt["original_bbox"]
    )
```

---

## 6. Ideal Segmentation: Mask Refinement

### 6.1 Refinement Pipeline Overview

```
+-----------------------------------------------------------------------------------+
|                        MASK REFINEMENT PIPELINE                                    |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|    RAW SAM MASK                                                                    |
|         |                                                                          |
|         v                                                                          |
|    +------------------+                                                            |
|    | 1. Clean Edges   |  Morphological operations (open/close)                    |
|    +------------------+                                                            |
|         |                                                                          |
|         v                                                                          |
|    +------------------+                                                            |
|    | 2. Fill Holes    |  Binary fill for internal gaps                            |
|    +------------------+                                                            |
|         |                                                                          |
|         v                                                                          |
|    +------------------+                                                            |
|    | 3. Keep Largest  |  Remove small disconnected components                     |
|    +------------------+                                                            |
|         |                                                                          |
|         v                                                                          |
|    +------------------+                                                            |
|    | 4. Smooth Border |  Gaussian blur + threshold for smooth edges               |
|    +------------------+                                                            |
|         |                                                                          |
|         v                                                                          |
|    REFINED MASK                                                                    |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

### 6.2 Refinement Operations

```
FUNCTION refine_mask(
    mask: np.ndarray,
    config: RefinementConfig
) -> np.ndarray:
    """
    Apply morphological operations to clean up SAM mask.
    """
    # Ensure binary mask (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Step 1: Morphological opening (remove small noise)
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (config.open_kernel_size, config.open_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Step 2: Morphological closing (fill small gaps)
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.close_kernel_size, config.close_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Step 3: Fill holes in the mask
    mask = fill_holes(mask)
    
    # Step 4: Keep only the largest connected component
    mask = keep_largest_component(mask)
    
    # Step 5: Smooth edges (optional)
    IF config.smooth_edges:
        mask = smooth_mask_edges(mask, config.smooth_sigma)
    
    RETURN mask


FUNCTION fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in the binary mask.
    """
    # Flood fill from edges
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with original
    RETURN mask | filled_inv


FUNCTION keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    Removes small artifacts and disconnected regions.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    IF num_labels <= 1:
        RETURN mask  # No components or only background
    
    # Find largest component (excluding background at label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only largest component
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    
    RETURN largest_mask


FUNCTION smooth_mask_edges(
    mask: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Apply Gaussian blur and re-threshold for smoother edges.
    """
    # Blur
    blurred = cv2.GaussianBlur(mask.astype(float), (0, 0), sigma)
    
    # Re-threshold at 50%
    smoothed = (blurred > 127.5).astype(np.uint8) * 255
    
    RETURN smoothed
```

### 6.3 Refinement Configuration

```yaml
# Mask refinement configuration
refinement:
  open_kernel_size: 3          # Remove noise smaller than 3px
  close_kernel_size: 5         # Fill gaps smaller than 5px
  fill_holes: true             # Fill internal holes
  keep_largest: true           # Keep only largest component
  smooth_edges: true           # Apply edge smoothing
  smooth_sigma: 1.5            # Gaussian sigma for smoothing
```

---

## 7. Background Removal (Black Background)

### 7.1 Apply Black Background

```
FUNCTION apply_black_background(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Set all non-masked pixels to solid black (0,0,0).
    Results in isolated face+hair on black background.
    """
    # Ensure mask is binary and matches image dimensions
    IF mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create output image
    result = np.zeros_like(image)
    
    # Copy only masked pixels
    mask_bool = mask > 0
    result[mask_bool] = image[mask_bool]
    
    RETURN result
```

### 7.2 Alternative: Transparent Background (PNG with Alpha)

For future flexibility, optionally save with transparency:

```
FUNCTION apply_transparent_background(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Convert to RGBA with transparent background.
    Useful for compositing in future HITL interface.
    """
    # Ensure RGB
    IF len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create RGBA image
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = mask  # Alpha channel from mask
    
    RETURN rgba
```

---

## 8. Face Cropping Strategy

### 8.1 Crop to Mask Bounding Box

```
FUNCTION crop_to_face(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Crop image and mask to the bounding box of the mask.
    Returns cropped image, mask, and crop metadata.
    """
    # Find mask bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    IF NOT np.any(rows) OR NOT np.any(cols):
        LOG_WARNING("Empty mask, cannot crop")
        RETURN image, mask, {"error": "empty_mask"}
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(mask.shape[0], y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(mask.shape[1], x_max + padding + 1)
    
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    crop_metadata = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "crop_width": x_max - x_min,
        "crop_height": y_max - y_min
    }
    
    RETURN cropped_image, cropped_mask, crop_metadata
```

### 8.2 Minimum Dimension Enforcement

AdaFace requires minimum 112x112 input. Ensure crops meet this threshold:

```
FUNCTION ensure_minimum_dimensions(
    image: np.ndarray,
    min_dimension: int = 112
) -> Tuple[np.ndarray, bool]:
    """
    Ensure image meets minimum dimension requirements.
    Returns (image, was_upscaled).
    """
    h, w = image.shape[:2]
    
    IF h >= min_dimension AND w >= min_dimension:
        RETURN image, False
    
    # Calculate scale factor
    scale = min_dimension / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Upscale using INTER_CUBIC for quality
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    LOG_DEBUG(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
    
    RETURN upscaled, True
```

---

## 9. Output I/O Strategy

### 9.1 Output Directory Structure

```
./output/
+-- masks/
|   +-- 20231027/                          # Date-based organization (YYYYMMDD)
|   |   +-- session_1/                     # Session grouping (from filename H)
|   |   |   +-- 202310271_face_001_a1b2c3.png
|   |   |   +-- 202310271_face_001_a1b2c3.json   # Metadata sidecar
|   |   |   +-- 202310271_face_002_d4e5f6.png
|   |   |   +-- 202310271_face_002_d4e5f6.json
|   |   |   +-- ...
|   |   +-- session_2/
|   |   |   +-- ...
|   +-- 20231028/
|   |   +-- ...
+-- metadata/
|   +-- processing_log.json                 # Overall processing log
|   +-- 20231027_session_1_manifest.json    # Session-level manifest
|   +-- 20231027_session_2_manifest.json
+-- debug/
    +-- segmentations/                      # Optional debug visualizations
        +-- 202310271_segmentation_overlay.jpg
```

### 9.2 Output File Naming Convention

**Pattern**: `{original_stem}_face_{index:03d}_{bbox_hash}.png`

| Component | Description | Example |
|-----------|-------------|---------|
| `original_stem` | Original filename without extension | `202310271` |
| `face` | Literal string for identification | `face` |
| `index` | Zero-padded face index (001-999) | `001` |
| `bbox_hash` | 6-char hash of original bbox coords | `a1b2c3` |

**Hash Calculation**:
```
FUNCTION compute_bbox_hash(bbox: np.ndarray) -> str:
    """
    Compute short hash of bounding box for unique identification.
    """
    # Create deterministic string from bbox
    bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
    
    # Hash and truncate
    hash_full = hashlib.md5(bbox_str.encode()).hexdigest()
    RETURN hash_full[:6]
```

### 9.3 Output Image Specifications

| Property | Value | Rationale |
|----------|-------|-----------|
| **Format** | PNG | Lossless, no compression artifacts |
| **Color Space** | RGB | Standard for downstream models |
| **Background** | Solid Black (0,0,0) | Clean mask for AdaFace |
| **Bit Depth** | 8-bit per channel | Standard |
| **Min Dimension** | 112px | AdaFace requirement |
| **Max Dimension** | Preserve original | No arbitrary downscaling |

### 9.4 Metadata Sidecar File

Each face image has a corresponding JSON metadata file:

```json
{
    "version": "1.0",
    "generated_at": "2026-03-28T14:30:00.000Z",
    "source": {
        "filename": "202310271.jpg",
        "date": "2023-10-27",
        "session": 1,
        "original_resolution": [1600, 1200]
    },
    "detection": {
        "detection_id": "202310271_001",
        "original_bbox": [234, 156, 312, 248],
        "expanded_bbox": [210, 110, 336, 268],
        "confidence": 0.92
    },
    "segmentation": {
        "sam_score": 0.95,
        "mask_area_pixels": 8432,
        "refinement_applied": true
    },
    "output": {
        "crop_bbox": [210, 110, 336, 268],
        "final_dimensions": [126, 158],
        "was_upscaled": false,
        "background_color": [0, 0, 0]
    },
    "quality_flags": {
        "low_resolution": false,
        "possible_occlusion": false,
        "segmentation_quality": "good"
    }
}
```

### 9.5 Session Manifest File

Aggregates all face outputs for a session:

```json
{
    "session_id": "20231027_session_1",
    "source_image": "202310271.jpg",
    "processing_timestamp": "2026-03-28T14:30:00.000Z",
    "total_faces_detected": 24,
    "total_faces_segmented": 22,
    "failed_segmentations": 2,
    "processing_time_seconds": 45.2,
    "faces": [
        {
            "output_file": "202310271_face_001_a1b2c3.png",
            "metadata_file": "202310271_face_001_a1b2c3.json",
            "confidence": 0.92,
            "sam_score": 0.95
        },
        ...
    ],
    "errors": [
        {
            "detection_id": "202310271_023",
            "error": "segmentation_failed",
            "details": "Empty mask returned"
        }
    ]
}
```

---

## 10. Complete Segmentation Pipeline

### 10.1 Full Stage Flow

```
FUNCTION run_segmentation_stage(
    detection_results: List[DetectionResult],
    config: SegmentationConfig
) -> List[SessionSegmentationResult]:
    """
    Full segmentation stage for all detected faces across all images.
    """
    all_results = []
    
    # Load SAM model
    predictor = load_segmentation_model(config)
    
    FOR det_result IN detection_results:
        LOG_INFO(f"Segmenting {det_result.image_metadata.filename}")
        
        # Load original image
        image = load_image(det_result.image_metadata.filepath)
        
        # Prepare SAM prompts from detections
        prompts = prepare_sam_prompts(
            det_result.detections,
            image.shape
        )
        
        # Segment all faces
        seg_results = segment_all_faces(predictor, image, prompts, config)
        
        # Post-process and save each face
        session_results = []
        FOR i, seg_result IN enumerate(seg_results):
            IF seg_result.mask IS None:
                LOG_WARNING(f"Skipping failed segmentation: {seg_result.detection_id}")
                CONTINUE
            
            # Refine mask
            refined_mask = refine_mask(seg_result.mask, config.refinement)
            
            # Apply black background
            masked_image = apply_black_background(image, refined_mask)
            
            # Crop to face
            cropped_image, cropped_mask, crop_meta = crop_to_face(
                masked_image, refined_mask, padding=5
            )
            
            # Ensure minimum dimensions
            final_image, was_upscaled = ensure_minimum_dimensions(
                cropped_image, config.min_dimension
            )
            
            # Generate output paths
            output_path, metadata_path = generate_output_paths(
                det_result.image_metadata,
                seg_result,
                i,
                config.output_dir
            )
            
            # Save image and metadata
            save_face_output(
                final_image,
                output_path,
                metadata_path,
                det_result,
                seg_result,
                crop_meta,
                was_upscaled
            )
            
            session_results.append(FaceOutput(
                path=output_path,
                metadata_path=metadata_path,
                confidence=seg_result.original_bbox["confidence"],
                sam_score=seg_result.score
            ))
        
        # Save session manifest
        save_session_manifest(det_result, session_results, config)
        
        all_results.append(SessionSegmentationResult(
            image_metadata=det_result.image_metadata,
            faces=session_results,
            processing_time=elapsed_time
        ))
    
    # Unload SAM model
    unload_segmentation_model(predictor)
    
    RETURN all_results
```

### 10.2 Save Functions

```
FUNCTION save_face_output(
    image: np.ndarray,
    output_path: Path,
    metadata_path: Path,
    det_result: DetectionResult,
    seg_result: SegmentationResult,
    crop_meta: Dict,
    was_upscaled: bool
) -> None:
    """
    Save face image and metadata to disk.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG image
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate metadata
    metadata = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "filename": det_result.image_metadata.filename,
            "date": det_result.image_metadata.date.isoformat(),
            "session": det_result.image_metadata.session,
            "original_resolution": [
                det_result.image_metadata.width,
                det_result.image_metadata.height
            ]
        },
        "detection": {
            "detection_id": seg_result.detection_id,
            "original_bbox": seg_result.original_bbox.tolist(),
            "expanded_bbox": seg_result.bbox.tolist(),
            "confidence": float(seg_result.original_confidence)
        },
        "segmentation": {
            "sam_score": float(seg_result.score),
            "mask_area_pixels": int(np.sum(seg_result.mask > 0)),
            "refinement_applied": True
        },
        "output": {
            "crop_bbox": [
                crop_meta["x_min"],
                crop_meta["y_min"],
                crop_meta["x_max"],
                crop_meta["y_max"]
            ],
            "final_dimensions": [image.shape[1], image.shape[0]],
            "was_upscaled": was_upscaled,
            "background_color": [0, 0, 0]
        },
        "quality_flags": {
            "low_resolution": image.shape[0] < 112 or image.shape[1] < 112,
            "possible_occlusion": seg_result.occlusion_score > 0.3,
            "segmentation_quality": assess_quality(seg_result.score)
        }
    }
    
    # Save metadata JSON
    WITH open(metadata_path, 'w') AS f:
        json.dump(metadata, f, indent=2)
    
    LOG_DEBUG(f"Saved: {output_path.name}")


FUNCTION generate_output_paths(
    image_metadata: ImageMetadata,
    seg_result: SegmentationResult,
    index: int,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Generate output file paths for face image and metadata.
    """
    # Parse date components
    date_str = image_metadata.date.strftime("%Y%m%d")
    session_str = f"session_{image_metadata.session}"
    
    # Generate bbox hash
    bbox_hash = compute_bbox_hash(seg_result.original_bbox)
    
    # Build filename
    filename_stem = f"{image_metadata.filename.stem}_face_{index+1:03d}_{bbox_hash}"
    
    # Build paths
    output_subdir = output_dir / "masks" / date_str / session_str
    image_path = output_subdir / f"{filename_stem}.png"
    metadata_path = output_subdir / f"{filename_stem}.json"
    
    RETURN image_path, metadata_path
```

---

## 11. Debug Visualization

### 11.1 Segmentation Overlay Visualization

```
FUNCTION create_segmentation_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    detections: List[Detection],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create debug visualization showing all segmentation masks overlaid on image.
    """
    overlay = image.copy()
    
    # Color palette for different faces
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    FOR i, (mask, det) IN enumerate(zip(masks, detections)):
        color = colors[i % len(colors)]
        
        # Create colored mask overlay
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        
        # Blend with original
        overlay = cv2.addWeighted(
            overlay, 1,
            mask_colored, alpha,
            0
        )
        
        # Draw detection bbox
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"#{i+1} ({det.confidence:.2f})"
        cv2.putText(overlay, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    RETURN overlay


FUNCTION save_debug_segmentation(
    overlay: np.ndarray,
    image_metadata: ImageMetadata,
    output_dir: Path
) -> None:
    """
    Save segmentation debug visualization.
    """
    debug_dir = output_dir / "debug" / "segmentations"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{image_metadata.filename.stem}_segmentation_overlay.jpg"
    output_path = debug_dir / filename
    
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    LOG_DEBUG(f"Saved debug visualization: {output_path}")
```

---

## 12. Error Handling

### 12.1 Error Categories

| Error | Cause | Recovery |
|-------|-------|----------|
| `CUDA OOM` | Too many faces/large image | Reduce batch size, process one-by-one |
| `Empty mask` | SAM failed to segment | Skip face, log warning, continue |
| `Low quality mask` | Ambiguous segmentation | Flag for HITL review |
| `Save failure` | Disk full/permissions | Log error, continue with other faces |

### 12.2 Graceful Degradation

```
FUNCTION safe_segment(
    predictor: SamPredictor,
    image: np.ndarray,
    prompts: List[Dict],
    config: SegmentationConfig
) -> List[SegmentationResult]:
    """
    Segment with error handling and fallbacks.
    """
    TRY:
        RETURN segment_all_faces(predictor, image, prompts, config)
    
    EXCEPT torch.cuda.OutOfMemoryError:
        LOG_WARNING("OOM during segmentation, switching to single-face mode")
        torch.cuda.empty_cache()
        
        # Process one face at a time
        results = []
        FOR prompt IN prompts:
            TRY:
                predictor.set_image(image)
                result = segment_single_face_no_embed(predictor, prompt, config)
                results.append(result)
                torch.cuda.empty_cache()
            EXCEPT Exception AS e:
                LOG_ERROR(f"Single face segmentation failed: {e}")
                results.append(SegmentationResult(mask=None, error=str(e)))
        
        RETURN results
    
    EXCEPT Exception AS e:
        LOG_ERROR(f"Segmentation failed: {e}")
        RETURN [SegmentationResult(mask=None, error=str(e)) FOR _ IN prompts]
```

---

## 13. Logging Specifications

```
# Segmentation stage log examples

INFO  | Loading MobileSAM model | checkpoint=mobile_sam.pt | device=cuda:0 | half=True
INFO  | Model loaded | VRAM=1.52GB | warmup_complete=True

INFO  | Segmenting image | filename=202310271.jpg | faces=24
DEBUG | Computing image embedding | resolution=1600x1200
DEBUG | Processing batch | faces=8 | batch=1/3

DEBUG | Generated mask | detection_id=202310271_001 | score=0.95 | area=8432px
DEBUG | Generated mask | detection_id=202310271_002 | score=0.91 | area=7856px
WARN  | Low quality mask | detection_id=202310271_015 | score=0.72 | flagged=HITL

DEBUG | Refining mask | detection_id=202310271_001 | operations=[open,close,fill,smooth]
DEBUG | Cropping face | detection_id=202310271_001 | crop_size=126x158 | upscaled=False

INFO  | Saved face output | path=masks/20231027/session_1/202310271_face_001_a1b2c3.png
INFO  | Session complete | filename=202310271.jpg | segmented=22/24 | time=45.2s

ERROR | Segmentation failed | detection_id=202310271_023 | error="Empty mask returned"
WARN  | Skipped face | detection_id=202310271_024 | reason="below_min_dimension"

INFO  | Unloading MobileSAM model | freed_vram=1.52GB
INFO  | Segmentation stage complete | total_images=5 | total_faces=112 | time=225s
```

---

## 14. Configuration Reference

```yaml
# Segmentation stage configuration (subset of pipeline_config.yaml)

segmentation:
  model:
    type: "mobilesam"
    checkpoint_path: "/app/models/mobilesam/mobile_sam.pt"
    device: "cuda:0"
    half: true                      # FP16 inference
  
  inference:
    multimask_output: true          # Generate 3 masks, pick best
    pred_iou_thresh: 0.88
    stability_score_thresh: 0.95
    batch_size: 8                   # Faces per batch
    
  prompting:
    use_expanded_bbox: true         # Expand bbox for hair
    expand_top_ratio: 0.45          # Extra expansion on top
    expand_bottom_ratio: 0.15
    expand_horizontal_ratio: 0.30
  
  refinement:
    enabled: true
    open_kernel_size: 3
    close_kernel_size: 5
    fill_holes: true
    keep_largest: true
    smooth_edges: true
    smooth_sigma: 1.5
  
  output:
    format: "png"
    min_dimension: 112              # AdaFace requirement
    background_color: [0, 0, 0]     # RGB black
    save_metadata: true
    save_manifest: true
    
  debug:
    save_overlays: true
    save_individual_masks: false
  
  memory:
    aggressive_cleanup: false       # Call empty_cache after each batch
```

---

*Document Version: 1.0*  
*Last Updated: 2026-03-28*  
*Author: CV Pipeline Planning*
