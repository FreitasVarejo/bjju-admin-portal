#!/usr/bin/env python3
"""
Full Pipeline Demo Script
==========================

Processes all images in media/ directory through:
1. Stage 2: Face Detection with YOLOv8n-face
2. Stage 3: Face Segmentation with MobileSAM

Outputs:
- Detection debug images in output/debug/detections/
- Segmented face crops in output/masks/{date}/session_{N}/
- Metadata JSON files alongside each face
- Summary report of processing statistics

Hardware: NVIDIA RTX 4060 (8GB VRAM)
Strategy: Sequential model loading (YOLOv8 -> unload -> MobileSAM)
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple
import gc
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cv_pipeline.stage2_detection.detector import (
    load_detection_model,
    unload_detection_model,
    process_image as detect_image,
    save_detection_debug
)
from cv_pipeline.stage2_detection.models import DetectionConfig
from cv_pipeline.stage3_segmentation.segmenter import (
    load_segmentation_model,
    unload_segmentation_model,
    process_image as segment_image,
    save_face_output
)
from cv_pipeline.stage3_segmentation.models import SegmentationConfig
from cv_pipeline.stage1_ingestion.models import ImageMetadata, ImageStatus
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

MEDIA_DIR = project_root / "media"
OUTPUT_DIR = project_root / "output"
MODEL_YOLO_PATH = project_root / "models" / "yolov8" / "yolov8n-face.pt"
MODEL_SAM_PATH = project_root / "models" / "mobilesam" / "mobile_sam.pt"

# Detection configuration
DETECTION_CONFIG = DetectionConfig(
    weights_path=MODEL_YOLO_PATH,
    conf=0.3,            # Lower threshold to catch more faces
    iou=0.45,
    max_det=100,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    min_confidence=0.3,
    min_face_width=30,
    min_face_height=30,
    verbose=False
)

# Segmentation configuration  
SEGMENTATION_CONFIG = SegmentationConfig(
    weights_path=MODEL_SAM_PATH,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    min_width=112,
    min_height=112,
    verbose=False
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_images() -> List[Path]:
    """Get all JPEG images from media directory, sorted by filename."""
    images = sorted(MEDIA_DIR.glob("*.jpeg"))
    logger.info(f"Found {len(images)} images in {MEDIA_DIR}")
    return images


def parse_image_date(image_path: Path) -> str:
    """Extract date from filename like 2026022401.jpeg -> 20260224"""
    stem = image_path.stem  # "2026022401"
    return stem[:8]  # "20260224"


def create_image_metadata(image_path: Path) -> ImageMetadata:
    """Create ImageMetadata from image file."""
    # Parse filename like "2026022401.jpeg" -> date=20260224, session=1
    stem = image_path.stem
    year = int(stem[:4])
    month = int(stem[4:6])
    day = int(stem[6:8])
    session = int(stem[8:]) if len(stem) > 8 else 1
    
    capture_date = datetime(year, month, day)
    
    # Get image dimensions
    import cv2
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2] if img is not None else (0, 0)
    
    # Create metadata
    metadata = ImageMetadata(
        original_filename=image_path.name,
        original_path=image_path,
        file_size_bytes=image_path.stat().st_size,
        capture_date=capture_date,
        session_number=session,
        original_width=width,
        original_height=height,
        original_aspect_ratio=width / height if height > 0 else 0,
        status=ImageStatus.VALID
    )
    
    return metadata


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ============================================================================
# STAGE 2: DETECTION
# ============================================================================

def run_detection_stage(image_paths: List[Path]) -> Tuple[Dict[Path, any], Dict]:
    """
    Run face detection on all images.
    
    Returns:
        - detection_results: Dict mapping image_path -> DetectionResult
        - stats: Dict with detection statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: FACE DETECTION")
    logger.info("=" * 80)
    
    # Load detection model
    logger.info(f"Loading YOLOv8 model from {MODEL_YOLO_PATH}")
    model = load_detection_model(DETECTION_CONFIG)
    
    detection_results = {}
    stats = {
        "total_images": len(image_paths),
        "successful": 0,
        "failed": 0,
        "total_faces": 0,
        "total_time": 0.0,
        "errors": []
    }
    
    # Process each image
    for idx, image_path in enumerate(image_paths, 1):
        logger.info(f"\n[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            start_time = time.time()
            
            # Create metadata
            metadata = create_image_metadata(image_path)
            
            # Detect faces
            result = detect_image(
                image_path=Path(image_path),
                image_metadata=metadata,
                model=model,
                config=DETECTION_CONFIG
            )
            
            elapsed = time.time() - start_time
            stats["total_time"] += elapsed
            
            # Save debug image if enabled
            if DETECTION_CONFIG.enable_debug and result.debug_image is not None:
                save_detection_debug(result, OUTPUT_DIR)
                logger.info(f"  Saved debug image to {DETECTION_CONFIG.debug_output_dir}")
            
            # Store result
            detection_results[image_path] = result
            stats["successful"] += 1
            stats["total_faces"] += len(result.detections)
            
            logger.success(
                f"  Detected {len(result.detections)} faces in {elapsed:.2f}s "
                f"({result.image_width}x{result.image_height})"
            )
            
        except Exception as e:
            logger.error(f"  Failed to process {image_path.name}: {e}")
            stats["failed"] += 1
            stats["errors"].append({
                "image": image_path.name,
                "error": str(e)
            })
            continue
    
    # Unload model and free VRAM
    logger.info("\nUnloading detection model...")
    unload_detection_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DETECTION STAGE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total faces detected: {stats['total_faces']}")
    logger.info(f"Average faces per image: {stats['total_faces'] / max(stats['successful'], 1):.1f}")
    logger.info(f"Total time: {format_time(stats['total_time'])}")
    logger.info(f"Average time per image: {stats['total_time'] / max(stats['successful'], 1):.2f}s")
    
    return detection_results, stats


# ============================================================================
# STAGE 3: SEGMENTATION
# ============================================================================

def run_segmentation_stage(detection_results: Dict[Path, any]) -> Dict:
    """
    Run face segmentation on detected faces.
    
    Args:
        detection_results: Dict mapping image_path -> DetectionResult
        
    Returns:
        stats: Dict with segmentation statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: FACE SEGMENTATION")
    logger.info("=" * 80)
    
    # Load segmentation model
    logger.info(f"Loading MobileSAM model from {MODEL_SAM_PATH}")
    model = load_segmentation_model(SEGMENTATION_CONFIG)
    
    stats = {
        "total_sessions": 0,
        "successful_sessions": 0,
        "failed_sessions": 0,
        "total_faces_input": 0,
        "total_faces_segmented": 0,
        "total_time": 0.0,
        "errors": []
    }
    
    # Group images by session (date)
    sessions_by_date: Dict[str, List[Tuple[Path, any]]] = {}
    for image_path, result in detection_results.items():
        if result is None or len(result.detections) == 0:
            continue
        
        date = parse_image_date(image_path)
        if date not in sessions_by_date:
            sessions_by_date[date] = []
        sessions_by_date[date].append((image_path, result))
    
    logger.info(f"Processing {len(sessions_by_date)} unique dates/sessions")
    
    # Process each session
    for session_idx, (date, session_data) in enumerate(sorted(sessions_by_date.items()), 1):
        logger.info(f"\n[Session {session_idx}/{len(sessions_by_date)}] Date: {date}")
        
        stats["total_sessions"] += 1
        
        try:
            start_time = time.time()
            
            # Create output directory for this session
            output_base = OUTPUT_DIR / "masks" / date / f"session_{session_idx}"
            output_base.mkdir(parents=True, exist_ok=True)
            
            # Process each image in the session
            total_faces = 0
            saved_count = 0
            
            for image_path, det_result in session_data:
                if det_result is None or len(det_result.detections) == 0:
                    continue
                
                total_faces += len(det_result.detections)
                stats["total_faces_input"] += len(det_result.detections)
                
                logger.info(f"  Processing {image_path.name} ({len(det_result.detections)} faces)")
                
                # Segment faces in this image
                seg_result = segment_image(
                    image_path=image_path,
                    det_result=det_result,
                    predictor=model,
                    config=SEGMENTATION_CONFIG,
                    output_dir=output_base
                )
                
                # Count successful segmentations
                saved_count += seg_result.success_count
            
            elapsed = time.time() - start_time
            stats["total_time"] += elapsed
            
            stats["successful_sessions"] += 1
            stats["total_faces_segmented"] += saved_count
            
            logger.success(
                f"  Session complete: {saved_count}/{total_faces} faces segmented in {elapsed:.2f}s"
            )
            logger.info(f"  Output: {output_base}")
            
        except Exception as e:
            logger.error(f"  Failed to process session {date}: {e}")
            stats["failed_sessions"] += 1
            stats["errors"].append({
                "session": date,
                "error": str(e)
            })
            continue
    
    # Unload model
    logger.info("\nUnloading segmentation model...")
    unload_segmentation_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SEGMENTATION STAGE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total sessions: {stats['total_sessions']}")
    logger.info(f"Successful: {stats['successful_sessions']}")
    logger.info(f"Failed: {stats['failed_sessions']}")
    logger.info(f"Total faces input: {stats['total_faces_input']}")
    logger.info(f"Total faces segmented: {stats['total_faces_segmented']}")
    logger.info(f"Success rate: {stats['total_faces_segmented'] / max(stats['total_faces_input'], 1) * 100:.1f}%")
    logger.info(f"Total time: {format_time(stats['total_time'])}")
    logger.info(f"Average time per session: {stats['total_time'] / max(stats['successful_sessions'], 1):.2f}s")
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("BJJ ADMIN PORTAL - FULL CV PIPELINE DEMO")
    logger.info("=" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Media directory: {MEDIA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"YOLOv8 model: {MODEL_YOLO_PATH}")
    logger.info(f"MobileSAM model: {MODEL_SAM_PATH}")
    logger.info(f"Device: {DETECTION_CONFIG.device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Verify paths exist
    if not MEDIA_DIR.exists():
        logger.error(f"Media directory not found: {MEDIA_DIR}")
        return
    
    if not MODEL_YOLO_PATH.exists():
        logger.error(f"YOLOv8 model not found: {MODEL_YOLO_PATH}")
        return
    
    if not MODEL_SAM_PATH.exists():
        logger.error(f"MobileSAM model not found: {MODEL_SAM_PATH}")
        return
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DETECTION_CONFIG.debug_output_dir.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_CONFIG.debug_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_paths = get_all_images()
    if not image_paths:
        logger.error("No images found in media directory")
        return
    
    # Start processing
    overall_start = time.time()
    
    # Stage 2: Detection
    detection_results, detection_stats = run_detection_stage(image_paths)
    
    # Stage 3: Segmentation
    segmentation_stats = run_segmentation_stage(detection_results)
    
    # Final summary
    total_time = time.time() - overall_start
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images processed: {detection_stats['successful']}/{detection_stats['total_images']}")
    logger.info(f"Total faces detected: {detection_stats['total_faces']}")
    logger.info(f"Total faces segmented: {segmentation_stats['total_faces_segmented']}")
    logger.info(f"Overall success rate: {segmentation_stats['total_faces_segmented'] / max(detection_stats['total_faces'], 1) * 100:.1f}%")
    logger.info(f"Total processing time: {format_time(total_time)}")
    logger.info(f"Average time per image: {total_time / max(detection_stats['successful'], 1):.2f}s")
    logger.info("\nOutput locations:")
    logger.info(f"  Detection debug images: {DETECTION_CONFIG.debug_output_dir}")
    logger.info(f"  Segmented faces: {OUTPUT_DIR / 'masks'}")
    logger.info(f"  Segmentation debug: {SEGMENTATION_CONFIG.debug_output_dir}")
    logger.info("=" * 80)
    
    # Report errors if any
    if detection_stats['errors']:
        logger.warning(f"\nDetection errors ({len(detection_stats['errors'])}):")
        for err in detection_stats['errors'][:5]:  # Show first 5
            logger.warning(f"  {err['image']}: {err['error']}")
    
    if segmentation_stats['errors']:
        logger.warning(f"\nSegmentation errors ({len(segmentation_stats['errors'])}):")
        for err in segmentation_stats['errors'][:5]:  # Show first 5
            logger.warning(f"  {err['session']}: {err['error']}")


if __name__ == "__main__":
    main()
