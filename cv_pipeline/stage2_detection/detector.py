"""
Face Detection Pipeline using YOLOv8-face

This module implements the face detection stage with support for:
- YOLOv8-face model loading/unloading
- Direct, scaled, and tiled inference
- Detection filtering and NMS
- Bounding box expansion for SAM
- VRAM-aware sequential model loading
"""

import gc
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from loguru import logger

from cv_pipeline.stage2_detection.models import (
    Detection,
    DetectionConfig,
    DetectionResult,
)
from cv_pipeline.stage1_ingestion.models import ImageMetadata
from cv_pipeline.utils.exceptions import PipelineException


class DetectionError(PipelineException):
    """Raised when detection fails."""
    pass


def load_detection_model(config: DetectionConfig) -> Any:
    """
    Load YOLOv8-face model with memory optimization.
    
    Args:
        config: Detection configuration
        
    Returns:
        Loaded YOLO model
        
    Raises:
        DetectionError: If model loading fails
    """
    try:
        # Import here to avoid circular dependencies
        from ultralytics import YOLO
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache before YOLO load")
        
        logger.info(f"Loading YOLOv8-face model | weights={config.weights_path} | device={config.device} | half={config.half}")
        
        # Load model
        model = YOLO(str(config.weights_path))
        
        # Move to GPU with FP16
        model.to(config.device)
        if config.half and torch.cuda.is_available():
            model.model.half()
            logger.debug("Enabled FP16 mode")
        
        # Warm-up inference (allocates CUDA memory)
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(
            dummy_input,
            imgsz=config.imgsz,
            verbose=False,
            device=config.device
        )
        logger.debug("Warm-up inference complete")
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"YOLO model loaded | VRAM: {allocated:.2f} GB")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLOv8-face model: {e}")
        raise DetectionError(f"Model loading failed: {e}")


def unload_detection_model(model: Any) -> None:
    """
    Explicitly unload YOLO model and free GPU memory.
    Critical for sequential model loading strategy.
    
    Args:
        model: YOLO model to unload
    """
    if model is None:
        return
    
    logger.debug("Unloading YOLO model")
    
    # Delete model reference
    del model
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Verify memory freed
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"YOLO model unloaded | Remaining VRAM: {allocated:.2f} GB")


def detect_faces(
    image: np.ndarray,
    model: Any,
    config: DetectionConfig
) -> List[Detection]:
    """
    Run face detection on a single image.
    
    Args:
        image: Input image (BGR format)
        model: Loaded YOLO model
        config: Detection configuration
        
    Returns:
        List of Detection objects
    """
    # Run inference
    results = model.predict(
        source=image,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        max_det=config.max_det,
        half=config.half,
        verbose=config.verbose,
        device=config.device
    )
    
    # Extract detections
    detections = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
            
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(boxes.conf[i].cpu().item())
            class_id = int(boxes.cls[i].cpu().item())
            
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id
            )
            detections.append(detection)
    
    logger.debug(f"Detected {len(detections)} faces")
    return detections


def detect_faces_tiled(
    image: np.ndarray,
    model: Any,
    config: DetectionConfig,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> List[Detection]:
    """
    Tiled inference for high-resolution images.
    Prevents memory issues and improves small face detection.
    
    Args:
        image: Input image (BGR format)
        model: Loaded YOLO model
        config: Detection configuration
        tile_size: Size of each tile (default from config)
        overlap: Overlap between tiles (default from config)
        
    Returns:
        List of Detection objects with merged results
    """
    tile_size = tile_size or config.tile_size
    overlap = overlap or config.overlap
    
    height, width = image.shape[:2]
    all_detections = []
    
    # Calculate tile grid
    stride = tile_size - overlap
    n_tiles_x = math.ceil((width - overlap) / stride)
    n_tiles_y = math.ceil((height - overlap) / stride)
    
    logger.debug(f"Processing {n_tiles_x * n_tiles_y} tiles ({n_tiles_x}x{n_tiles_y})")
    
    for row in range(n_tiles_y):
        for col in range(n_tiles_x):
            # Calculate tile coordinates
            x1 = col * stride
            y1 = row * stride
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Pad if necessary (edge tiles)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            # Run detection on tile
            tile_detections = detect_faces(tile, model, config)
            
            # Transform coordinates to original image space
            for det in tile_detections:
                det.bbox[0] += x1  # x1
                det.bbox[1] += y1  # y1
                det.bbox[2] += x1  # x2
                det.bbox[3] += y1  # y2
                all_detections.append(det)
    
    # Merge overlapping detections across tiles
    merged_detections = apply_nms(all_detections, iou_threshold=0.5)
    
    logger.debug(f"Tiled detection: {len(all_detections)} raw -> {len(merged_detections)} merged")
    
    return merged_detections


def detect_faces_auto(
    image: np.ndarray,
    model: Any,
    config: DetectionConfig
) -> List[Detection]:
    """
    Automatically choose inference strategy based on image size.
    
    Args:
        image: Input image (BGR format)
        model: Loaded YOLO model
        config: Detection configuration
        
    Returns:
        List of Detection objects
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)
    
    if max_dim <= config.direct_inference_max:
        logger.debug(f"Using direct inference (max_dim={max_dim})")
        return detect_faces(image, model, config)
    
    elif max_dim <= config.scaled_inference_max:
        logger.debug(f"Using scaled inference (max_dim={max_dim})")
        scale = config.direct_inference_max / max_dim
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        detections = detect_faces(scaled, model, config)
        
        # Scale bboxes back to original resolution
        for det in detections:
            det.bbox = det.bbox / scale
            det.scale_detected = scale
        
        return detections
    
    else:
        logger.debug(f"Using tiled inference (max_dim={max_dim})")
        return detect_faces_tiled(image, model, config)


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.45
) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        detections: List of detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Convert to tensors for NMS
    boxes = torch.tensor([d.bbox for d in detections])
    scores = torch.tensor([d.confidence for d in detections])
    
    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    return [detections[i] for i in keep_indices.tolist()]


def filter_detections(
    detections: List[Detection],
    config: DetectionConfig
) -> Tuple[List[Detection], List[Detection]]:
    """
    Filter detections based on quality criteria.
    
    Args:
        detections: List of detections
        config: Detection configuration
        
    Returns:
        Tuple of (accepted, rejected) detections
    """
    accepted = []
    rejected = []
    
    for det in detections:
        rejection_reason = None
        
        # Filter 1: Confidence threshold
        if det.confidence < config.min_confidence:
            rejection_reason = f"low_confidence:{det.confidence:.2f}"
        
        # Filter 2: Minimum size
        elif det.width < config.min_face_width or det.height < config.min_face_height:
            rejection_reason = f"too_small:{det.width:.0f}x{det.height:.0f}"
        
        # Filter 3: Aspect ratio sanity
        elif det.aspect_ratio < config.min_aspect_ratio or det.aspect_ratio > config.max_aspect_ratio:
            rejection_reason = f"bad_aspect:{det.aspect_ratio:.2f}"
        
        # Filter 4: Maximum size
        elif det.width > config.max_face_width or det.height > config.max_face_height:
            rejection_reason = f"too_large:{det.width:.0f}x{det.height:.0f}"
        
        if rejection_reason:
            det.rejection_reason = rejection_reason
            rejected.append(det)
            logger.debug(f"Rejected detection: {rejection_reason}")
        else:
            accepted.append(det)
    
    logger.info(f"Filtering: {len(accepted)} accepted, {len(rejected)} rejected")
    return accepted, rejected


def expand_bbox_for_segmentation(
    bbox: np.ndarray,
    config: DetectionConfig,
    image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Expand bounding box to include hair and provide context for SAM.
    
    Args:
        bbox: Original bounding box [x1, y1, x2, y2]
        config: Detection configuration
        image_shape: Image shape (height, width) for clipping
        
    Returns:
        Expanded bounding box
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Asymmetric expansion (more on top for hair)
    expand_top = height * config.bbox_expand_top_ratio
    expand_bottom = height * config.bbox_expand_bottom_ratio
    expand_horizontal = width * config.bbox_expand_horizontal_ratio
    
    new_bbox = np.array([
        x1 - expand_horizontal,
        y1 - expand_top,
        x2 + expand_horizontal,
        y2 + expand_bottom
    ])
    
    # Clip to image boundaries
    if image_shape:
        img_h, img_w = image_shape[:2]
        new_bbox = np.array([
            max(0, new_bbox[0]),
            max(0, new_bbox[1]),
            min(img_w, new_bbox[2]),
            min(img_h, new_bbox[3])
        ])
    
    return new_bbox


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    # Intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1e-6)


def assess_occlusion(
    detection: Detection,
    all_detections: List[Detection]
) -> float:
    """
    Estimate occlusion level based on overlapping detections.
    
    Args:
        detection: Detection to assess
        all_detections: All detections in the image
        
    Returns:
        Occlusion score (0 = no occlusion, 1 = fully occluded)
    """
    occlusion_score = 0.0
    
    for other in all_detections:
        if other is detection:
            continue
        
        # Calculate IoU
        iou = calculate_iou(detection.bbox, other.bbox)
        
        # Check if current detection is below/behind another
        if iou > 0.1:
            if detection.center[1] > other.center[1]:  # Below in image
                # Lower face might be occluded by upper body of front person
                occlusion_score = max(occlusion_score, iou)
    
    return occlusion_score


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    rejected: Optional[List[Detection]] = None,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Draw detection bounding boxes on image for debugging.
    
    Args:
        image: Input image (BGR format)
        detections: Accepted detections
        rejected: Rejected detections (optional)
        show_landmarks: Whether to show landmarks
        
    Returns:
        Visualization image
    """
    vis_image = image.copy()
    
    # Draw accepted detections in green
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label with confidence
        label = f"{i+1}: {det.confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw expanded bbox in yellow
        if det.bbox_expanded is not None:
            ex1, ey1, ex2, ey2 = det.bbox_expanded.astype(int)
            cv2.rectangle(vis_image, (ex1, ey1), (ex2, ey2), (0, 255, 255), 1)
        
        # Draw landmarks if available
        if show_landmarks and det.landmarks is not None:
            for point in det.landmarks:
                cv2.circle(vis_image, tuple(point.astype(int)), 2, (255, 0, 0), -1)
    
    # Draw rejected detections in red
    if rejected:
        for det in rejected:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            # Label with rejection reason
            if det.rejection_reason:
                cv2.putText(vis_image, det.rejection_reason[:20], (x1, y2+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return vis_image


def process_image(
    image_path: Path,
    image_metadata: ImageMetadata,
    model: Any,
    config: DetectionConfig
) -> DetectionResult:
    """
    Process a single image through the detection pipeline.
    
    Args:
        image_path: Path to input image
        image_metadata: Image metadata from Stage 1
        model: Loaded YOLO model
        config: Detection configuration
        
    Returns:
        DetectionResult object
    """
    start_time = time.time()
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise DetectionError(f"Failed to load image: {image_path}")
        
        logger.debug(f"Processing image | filename={image_metadata.original_filename} | resolution={image.shape[1]}x{image.shape[0]}")
        
        # Run detection with auto strategy
        detections = detect_faces_auto(image, model, config)
        
        # Apply NMS (in case of tiled inference)
        detections = apply_nms(detections, config.iou)
        
        # Filter detections
        accepted, rejected = filter_detections(detections, config)
        
        # Assign detection IDs and expand bboxes
        image_stem = Path(image_metadata.original_filename).stem
        for i, det in enumerate(accepted):
            det.detection_id = f"{image_stem}_{i+1:03d}"
            det.bbox_expanded = expand_bbox_for_segmentation(
                det.bbox,
                config,
                image.shape[:2]
            )
            # Assess occlusion
            det.occlusion_score = assess_occlusion(det, accepted)
        
        # Calculate processing time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Determine inference method
        max_dim = max(image.shape[:2])
        if max_dim <= config.direct_inference_max:
            method = "direct"
        elif max_dim <= config.scaled_inference_max:
            method = "scaled"
        else:
            method = "tiled"
        
        # Create debug visualization
        debug_image = None
        if config.save_debug_visualizations:
            debug_image = draw_detections(image, accepted, rejected)
        
        # Create result
        result = DetectionResult(
            image_metadata=image_metadata,
            detections=accepted,
            rejected=rejected,
            inference_time_ms=inference_time_ms,
            detection_count=len(accepted),
            rejection_count=len(rejected),
            inference_method=method,
            debug_image=debug_image
        )
        
        logger.info(f"Detection complete | faces={len(accepted)} | rejected={len(rejected)} | time={inference_time_ms:.1f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed for {image_metadata.original_filename}: {e}")
        return DetectionResult(
            image_metadata=image_metadata,
            detections=[],
            rejected=[],
            error=str(e)
        )


def save_detection_debug(
    result: DetectionResult,
    output_dir: Path
) -> None:
    """
    Save detection results for debugging and verification.
    
    Args:
        result: Detection result
        output_dir: Output directory
    """
    # Create output directory
    debug_dir = output_dir / "debug" / "detections"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization image
    if result.debug_image is not None:
        vis_path = debug_dir / f"{Path(result.image_metadata.original_filename).stem}_detections.jpg"
        cv2.imwrite(str(vis_path), result.debug_image)
        logger.debug(f"Saved debug visualization: {vis_path}")
    
    # Save detection data as JSON
    import json
    json_path = debug_dir / f"{Path(result.image_metadata.original_filename).stem}_detections.json"
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.debug(f"Saved detection metadata: {json_path}")
