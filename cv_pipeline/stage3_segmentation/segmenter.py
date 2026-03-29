"""
Face Segmentation Pipeline using MobileSAM

This module implements the face segmentation stage with support for:
- MobileSAM model loading/unloading
- Bounding box prompting from YOLO detections
- Mask refinement with morphological operations
- Background removal and cropping
- Output saving with metadata
- VRAM-aware sequential model loading
"""

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from loguru import logger
from datetime import datetime

from cv_pipeline.stage3_segmentation.models import (
    SegmentationConfig,
    SegmentationResult,
    FaceOutput,
    SessionSegmentationResult,
)
from cv_pipeline.stage2_detection.models import Detection, DetectionResult
from cv_pipeline.utils.exceptions import PipelineException


class SegmentationError(PipelineException):
    """Raised when segmentation fails."""
    pass


def load_segmentation_model(config: SegmentationConfig) -> Any:
    """
    Load MobileSAM model after YOLO has been unloaded.
    
    Args:
        config: Segmentation configuration
        
    Returns:
        SAM predictor object
        
    Raises:
        SegmentationError: If model loading fails
    """
    try:
        # Import SAM dependencies
        from mobile_sam import sam_model_registry, SamPredictor
        
        # Verify YOLO is unloaded
        if torch.cuda.is_available():
            current_vram = torch.cuda.memory_allocated() / 1024**3
            if current_vram > 0.5:
                logger.warning(f"High VRAM before SAM load: {current_vram:.2f}GB")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        logger.info(f"Loading MobileSAM model | checkpoint={config.checkpoint_path} | device={config.device} | half={config.half}")
        
        # Load MobileSAM
        sam = sam_model_registry[config.model_type](checkpoint=str(config.checkpoint_path))
        
        # Move to GPU with FP16
        sam.to(config.device)
        if config.half and torch.cuda.is_available():
            sam = sam.half()
            logger.debug("Enabled FP16 mode")
        
        # Create predictor wrapper
        predictor = SamPredictor(sam)
        
        # Warm-up inference
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        predictor.set_image(dummy_image)
        dummy_box = np.array([100, 100, 200, 200])
        _ = predictor.predict(box=dummy_box, multimask_output=True)
        logger.debug("Warm-up inference complete")
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"MobileSAM loaded | VRAM: {allocated:.2f} GB")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to load MobileSAM model: {e}")
        raise SegmentationError(f"Model loading failed: {e}")


def unload_segmentation_model(predictor: Any) -> None:
    """
    Explicitly unload SAM model and free GPU memory.
    
    Args:
        predictor: SAM predictor to unload
    """
    if predictor is None:
        return
    
    logger.debug("Unloading MobileSAM model")
    
    # Delete predictor and model references
    if hasattr(predictor, 'model'):
        del predictor.model
    del predictor
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Verify memory freed
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"MobileSAM unloaded | Remaining VRAM: {allocated:.2f} GB")


def prepare_sam_prompts(
    detections: List[Detection],
    image_shape: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """
    Convert YOLO bounding boxes to SAM prompt format.
    Uses expanded bboxes to capture hair region.
    
    Args:
        detections: List of detections from YOLO
        image_shape: Image shape (height, width)
        
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    
    for det in detections:
        # Use pre-expanded bbox from detection stage
        bbox = det.bbox_expanded if det.bbox_expanded is not None else det.bbox
        
        # Ensure bbox is within image boundaries
        img_h, img_w = image_shape[:2]
        bbox = np.array([
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(img_w, bbox[2]),
            min(img_h, bbox[3])
        ])
        
        prompts.append({
            'box': bbox,
            'detection_id': det.detection_id,
            'original_bbox': det.bbox,
            'confidence': det.confidence,
            'occlusion_score': det.occlusion_score
        })
    
    return prompts


def segment_single_face(
    predictor: Any,
    prompt: Dict[str, Any],
    config: SegmentationConfig,
    compute_embedding: bool = True
) -> SegmentationResult:
    """
    Generate segmentation mask for a single face using box prompt.
    
    Args:
        predictor: SAM predictor
        prompt: Prompt dictionary with bbox and metadata
        config: Segmentation configuration
        compute_embedding: Whether to compute image embedding (first call only)
        
    Returns:
        SegmentationResult object
    """
    try:
        # Generate masks with box prompt
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=prompt['box'],
            multimask_output=config.multimask_output
        )
        
        # Select best mask based on IoU score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        logger.debug(f"Generated mask | detection_id={prompt['detection_id']} | score={best_score:.3f}")
        
        return SegmentationResult(
            mask=best_mask,
            score=float(best_score),
            detection_id=prompt['detection_id'],
            bbox=prompt['box'],
            original_bbox=prompt['original_bbox'],
            original_confidence=prompt['confidence'],
            occlusion_score=prompt['occlusion_score'],
            mask_area_pixels=int(np.sum(best_mask > 0))
        )
        
    except Exception as e:
        logger.warning(f"Segmentation failed for {prompt['detection_id']}: {e}")
        return SegmentationResult(
            mask=None,
            score=0.0,
            detection_id=prompt['detection_id'],
            error=str(e)
        )


def segment_all_faces(
    predictor: Any,
    image: np.ndarray,
    prompts: List[Dict[str, Any]],
    config: SegmentationConfig
) -> List[SegmentationResult]:
    """
    Segment all faces in an image efficiently.
    Image embedding is computed once and reused.
    
    Args:
        predictor: SAM predictor
        image: Input image (RGB format)
        prompts: List of prompt dictionaries
        config: Segmentation configuration
        
    Returns:
        List of SegmentationResult objects
    """
    results = []
    
    # Compute image embedding ONCE
    logger.debug(f"Computing image embedding | resolution={image.shape[1]}x{image.shape[0]}")
    predictor.set_image(image)
    
    # Process faces in batches for memory efficiency
    batch_size = config.batch_size
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        logger.debug(f"Processing batch | faces={len(batch_prompts)} | batch={i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        for prompt in batch_prompts:
            result = segment_single_face(
                predictor,
                prompt,
                config,
                compute_embedding=False  # Already computed
            )
            results.append(result)
        
        # Optional: Clear intermediate tensors
        if config.aggressive_cleanup:
            torch.cuda.empty_cache()
    
    successful = sum(1 for r in results if r.mask is not None)
    logger.info(f"Segmented {successful}/{len(results)} faces successfully")
    
    return results


def refine_mask(
    mask: np.ndarray,
    config: SegmentationConfig
) -> np.ndarray:
    """
    Apply morphological operations to clean up SAM mask.
    
    Args:
        mask: Raw binary mask from SAM
        config: Segmentation configuration
        
    Returns:
        Refined binary mask
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
    if config.fill_holes:
        mask = fill_holes(mask)
    
    # Step 4: Keep only the largest connected component
    if config.keep_largest:
        mask = keep_largest_component(mask)
    
    # Step 5: Smooth edges (optional)
    if config.smooth_edges:
        mask = smooth_mask_edges(mask, config.smooth_sigma)
    
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in the binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with holes filled
    """
    # Flood fill from edges
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with original
    return mask | filled_inv


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    Removes small artifacts and disconnected regions.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only largest component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    if num_labels <= 1:
        return mask  # No components or only background
    
    # Find largest component (excluding background at label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only largest component
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    
    return largest_mask


def smooth_mask_edges(
    mask: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Apply Gaussian blur and re-threshold for smoother edges.
    
    Args:
        mask: Binary mask
        sigma: Gaussian sigma for smoothing
        
    Returns:
        Smoothed binary mask
    """
    # Blur
    blurred = cv2.GaussianBlur(mask.astype(float), (0, 0), sigma)
    
    # Re-threshold at 50%
    smoothed = (blurred > 127.5).astype(np.uint8) * 255
    
    return smoothed


def apply_black_background(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Set all non-masked pixels to solid black (0,0,0).
    Results in isolated face+hair on black background.
    
    Args:
        image: Input image (RGB or BGR)
        mask: Binary mask
        
    Returns:
        Image with black background
    """
    # Ensure mask matches image dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    # Create output image
    result = np.zeros_like(image)
    
    # Copy only masked pixels
    mask_bool = mask > 0
    result[mask_bool] = image[mask_bool]
    
    return result


def crop_to_face(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Crop image and mask to the bounding box of the mask.
    
    Args:
        image: Input image
        mask: Binary mask
        padding: Padding around mask bbox
        
    Returns:
        Tuple of (cropped_image, cropped_mask, crop_metadata)
    """
    # Find mask bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        logger.warning("Empty mask, cannot crop")
        return image, mask, {"error": "empty_mask"}
    
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
        'x_min': int(x_min),
        'y_min': int(y_min),
        'x_max': int(x_max),
        'y_max': int(y_max),
        'crop_width': int(x_max - x_min),
        'crop_height': int(y_max - y_min)
    }
    
    return cropped_image, cropped_mask, crop_metadata


def ensure_minimum_dimensions(
    image: np.ndarray,
    min_dimension: int = 112
) -> Tuple[np.ndarray, bool]:
    """
    Ensure image meets minimum dimension requirements.
    
    Args:
        image: Input image
        min_dimension: Minimum required dimension
        
    Returns:
        Tuple of (image, was_upscaled)
    """
    h, w = image.shape[:2]
    
    if h >= min_dimension and w >= min_dimension:
        return image, False
    
    # Calculate scale factor
    scale = min_dimension / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Upscale using INTER_CUBIC for quality
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    logger.debug(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
    
    return upscaled, True


def compute_bbox_hash(bbox: np.ndarray) -> str:
    """
    Compute short hash of bounding box for unique identification.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        6-character hash string
    """
    # Create deterministic string from bbox
    bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
    
    # Hash and truncate
    hash_full = hashlib.md5(bbox_str.encode()).hexdigest()
    return hash_full[:6]


def assess_quality(score: float) -> str:
    """
    Assess segmentation quality based on SAM score.
    
    Args:
        score: SAM IoU score
        
    Returns:
        Quality rating: "good", "fair", or "poor"
    """
    if score >= 0.90:
        return "good"
    elif score >= 0.75:
        return "fair"
    else:
        return "poor"


def generate_output_paths(
    image_metadata: Any,
    seg_result: SegmentationResult,
    index: int,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Generate output file paths for face image and metadata.
    
    Args:
        image_metadata: ImageMetadata from Stage 1
        seg_result: SegmentationResult
        index: Face index in image
        output_dir: Base output directory
        
    Returns:
        Tuple of (image_path, metadata_path)
    """
    # Parse date components
    date_str = image_metadata.capture_date.strftime("%Y%m%d")
    session_str = f"session_{image_metadata.session_number}"
    
    # Generate bbox hash
    bbox_hash = compute_bbox_hash(seg_result.original_bbox)
    
    # Build filename
    filename_stem = f"{Path(image_metadata.original_filename).stem}_face_{index+1:03d}_{bbox_hash}"
    
    # Build paths
    output_subdir = output_dir / "masks" / date_str / session_str
    image_path = output_subdir / f"{filename_stem}.png"
    metadata_path = output_subdir / f"{filename_stem}.json"
    
    return image_path, metadata_path


def save_face_output(
    image: np.ndarray,
    output_path: Path,
    metadata_path: Path,
    det_result: DetectionResult,
    seg_result: SegmentationResult,
    crop_meta: Dict[str, Any],
    was_upscaled: bool,
    config: SegmentationConfig
) -> None:
    """
    Save face image and metadata to disk.
    
    Args:
        image: Face image (RGB)
        output_path: Output image path
        metadata_path: Output metadata path
        det_result: DetectionResult from Stage 2
        seg_result: SegmentationResult
        crop_meta: Crop metadata dictionary
        was_upscaled: Whether image was upscaled
        config: Segmentation configuration
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG image (convert RGB to BGR for cv2)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Generate metadata
    if config.save_metadata:
        metadata = {
            'version': '1.0',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'source': {
                'filename': det_result.image_metadata.original_filename,
                'date': det_result.image_metadata.capture_date.isoformat(),
                'session': det_result.image_metadata.session_number,
                'original_resolution': [
                    det_result.image_metadata.original_width,
                    det_result.image_metadata.original_height
                ]
            },
            'detection': {
                'detection_id': seg_result.detection_id,
                'original_bbox': seg_result.original_bbox.tolist(),
                'expanded_bbox': seg_result.bbox.tolist(),
                'confidence': float(seg_result.original_confidence)
            },
            'segmentation': {
                'sam_score': float(seg_result.score),
                'mask_area_pixels': seg_result.mask_area_pixels,
                'refinement_applied': config.refinement_enabled
            },
            'output': {
                'crop_bbox': [
                    crop_meta['x_min'],
                    crop_meta['y_min'],
                    crop_meta['x_max'],
                    crop_meta['y_max']
                ],
                'final_dimensions': [image.shape[1], image.shape[0]],
                'was_upscaled': was_upscaled,
                'background_color': list(config.background_color)
            },
            'quality_flags': {
                'low_resolution': image.shape[0] < config.min_dimension or image.shape[1] < config.min_dimension,
                'possible_occlusion': seg_result.occlusion_score > 0.3,
                'segmentation_quality': assess_quality(seg_result.score)
            }
        }
        
        # Save metadata JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.debug(f"Saved: {output_path.name}")


def save_session_manifest(
    det_result: DetectionResult,
    session_result: SessionSegmentationResult,
    output_dir: Path
) -> None:
    """
    Save session manifest file with all face outputs.
    
    Args:
        det_result: DetectionResult from Stage 2
        session_result: SessionSegmentationResult
        output_dir: Base output directory
    """
    # Create metadata directory
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate manifest filename
    date_str = det_result.image_metadata.capture_date.strftime("%Y%m%d")
    session_str = f"session_{det_result.image_metadata.session_number}"
    manifest_path = metadata_dir / f"{date_str}_{session_str}_manifest.json"
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(session_result.to_dict(), f, indent=2)
    
    logger.debug(f"Saved session manifest: {manifest_path}")


def create_segmentation_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    detections: List[Detection],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create debug visualization showing all segmentation masks overlaid on image.
    
    Args:
        image: Input image (RGB)
        masks: List of binary masks
        detections: List of detections
        alpha: Overlay transparency
        
    Returns:
        Overlay visualization image
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
    
    for i, (mask, det) in enumerate(zip(masks, detections)):
        if mask is None:
            continue
            
        color = colors[i % len(colors)]
        
        # Create colored mask overlay
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        
        # Blend with original
        overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
        
        # Draw detection bbox
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"#{i+1} ({det.confidence:.2f})"
        cv2.putText(overlay, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return overlay


def save_debug_segmentation(
    overlay: np.ndarray,
    image_metadata: Any,
    output_dir: Path
) -> None:
    """
    Save segmentation debug visualization.
    
    Args:
        overlay: Overlay visualization image
        image_metadata: ImageMetadata from Stage 1
        output_dir: Base output directory
    """
    debug_dir = output_dir / "debug" / "segmentations"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{Path(image_metadata.original_filename).stem}_segmentation_overlay.jpg"
    output_path = debug_dir / filename
    
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.debug(f"Saved debug visualization: {output_path}")


def process_image(
    image_path: Path,
    det_result: DetectionResult,
    predictor: Any,
    config: SegmentationConfig,
    output_dir: Path
) -> SessionSegmentationResult:
    """
    Process a single image through the segmentation pipeline.
    
    Args:
        image_path: Path to input image
        det_result: DetectionResult from Stage 2
        predictor: Loaded SAM predictor
        config: Segmentation configuration
        output_dir: Base output directory
        
    Returns:
        SessionSegmentationResult
    """
    start_time = time.time()
    
    logger.info(f"Segmenting image | filename={det_result.image_metadata.original_filename} | faces={det_result.detection_count}")
    
    # Load original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise SegmentationError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare SAM prompts from detections
    prompts = prepare_sam_prompts(det_result.detections, image_rgb.shape[:2])
    
    # Segment all faces
    seg_results = segment_all_faces(predictor, image_rgb, prompts, config)
    
    # Create session result
    session_result = SessionSegmentationResult(
        image_metadata=det_result.image_metadata
    )
    
    # Post-process and save each face
    masks_for_debug = []
    for i, seg_result in enumerate(seg_results):
        if seg_result.mask is None:
            logger.warning(f"Skipping failed segmentation: {seg_result.detection_id}")
            session_result.add_error(
                seg_result.detection_id,
                seg_result.error or "unknown_error",
                "Mask generation failed"
            )
            continue
        
        # Refine mask
        if config.refinement_enabled:
            refined_mask = refine_mask(seg_result.mask, config)
        else:
            refined_mask = (seg_result.mask > 0).astype(np.uint8) * 255
        
        masks_for_debug.append(refined_mask)
        
        # Apply black background
        masked_image = apply_black_background(image_rgb, refined_mask)
        
        # Crop to face
        cropped_image, cropped_mask, crop_meta = crop_to_face(
            masked_image,
            refined_mask,
            padding=config.crop_padding
        )
        
        if "error" in crop_meta:
            logger.warning(f"Skipping face with empty mask: {seg_result.detection_id}")
            session_result.add_error(seg_result.detection_id, "empty_mask", "Mask produced no pixels")
            continue
        
        # Update seg_result with crop info
        seg_result.crop_bbox = (
            crop_meta['x_min'],
            crop_meta['y_min'],
            crop_meta['x_max'],
            crop_meta['y_max']
        )
        seg_result.crop_width = crop_meta['crop_width']
        seg_result.crop_height = crop_meta['crop_height']
        
        # Ensure minimum dimensions
        final_image, was_upscaled = ensure_minimum_dimensions(
            cropped_image,
            config.min_dimension
        )
        seg_result.was_upscaled = was_upscaled
        seg_result.segmentation_quality = assess_quality(seg_result.score)
        
        # Generate output paths
        output_path, metadata_path = generate_output_paths(
            det_result.image_metadata,
            seg_result,
            i,
            output_dir
        )
        
        # Save image and metadata
        save_face_output(
            final_image,
            output_path,
            metadata_path,
            det_result,
            seg_result,
            crop_meta,
            was_upscaled,
            config
        )
        
        # Add to session results
        face_output = FaceOutput(
            path=output_path,
            metadata_path=metadata_path,
            confidence=seg_result.original_confidence,
            sam_score=seg_result.score,
            detection_id=seg_result.detection_id
        )
        session_result.add_face(face_output)
    
    # Calculate processing time
    session_result.processing_time_seconds = time.time() - start_time
    
    # Save session manifest
    if config.save_manifest:
        save_session_manifest(det_result, session_result, output_dir)
    
    # Create debug visualization
    if config.save_overlays and len(masks_for_debug) > 0:
        overlay = create_segmentation_overlay(
            image_rgb,
            masks_for_debug,
            det_result.detections,
            alpha=0.5
        )
        save_debug_segmentation(overlay, det_result.image_metadata, output_dir)
    
    logger.info(f"Session complete | filename={det_result.image_metadata.original_filename} | segmented={len(session_result.faces)}/{det_result.detection_count} | time={session_result.processing_time_seconds:.1f}s")
    
    return session_result
