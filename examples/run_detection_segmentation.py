#!/usr/bin/env python3
"""
Example: Running Stage 2 (Detection) and Stage 3 (Segmentation)

This script demonstrates how to run the face detection and segmentation pipeline
with proper VRAM management and sequential model loading.

Usage:
    python examples/run_detection_segmentation.py --input-dir ./data/preprocessed --output-dir ./output
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from cv_pipeline.stage2_detection.models import DetectionConfig
from cv_pipeline.stage2_detection.detector import (
    load_detection_model,
    unload_detection_model,
    process_image as detect_faces_in_image,
    save_detection_debug,
)

from cv_pipeline.stage3_segmentation.models import SegmentationConfig
from cv_pipeline.stage3_segmentation.segmenter import (
    load_segmentation_model,
    unload_segmentation_model,
    process_image as segment_faces_in_image,
)

from cv_pipeline.stage1_ingestion.models import ImageMetadata, ImageStatus
from datetime import datetime


def create_mock_image_metadata(image_path: Path) -> ImageMetadata:
    """
    Create mock ImageMetadata for testing.
    In production, this would come from Stage 1.
    """
    import cv2
    
    # Parse filename (format: YYYYMMDDN.jpg where N is session number)
    filename = image_path.stem
    
    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Extract date and session from filename
    # Example: 202310271.jpg -> date=2023-10-27, session=1
    try:
        date_str = filename[:8]  # YYYYMMDD
        session_num = int(filename[8]) if len(filename) > 8 else 1
        capture_date = datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        logger.warning(f"Could not parse date from filename {filename}, using current date")
        capture_date = datetime.now()
        session_num = 1
    
    return ImageMetadata(
        original_filename=image_path.name,
        original_path=image_path,
        file_size_bytes=image_path.stat().st_size,
        capture_date=capture_date,
        session_number=session_num,
        original_width=width,
        original_height=height,
        original_aspect_ratio=width / height,
        processed_path=image_path,  # Using same path for this example
        processed_width=width,
        processed_height=height,
        status=ImageStatus.PREPROCESSED,
    )


def run_pipeline(input_dir: Path, output_dir: Path, debug: bool = True):
    """
    Run the complete detection and segmentation pipeline.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        debug: Whether to save debug visualizations
    """
    # Configure logging
    logger.add(
        output_dir / "logs" / "pipeline_{time}.log",
        rotation="100 MB",
        level="DEBUG" if debug else "INFO"
    )
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Stage 2 Configuration
    detection_config = DetectionConfig(
        weights_path=Path("/app/models/yolov8/yolov8n-face.pt"),
        device="cuda:0",
        half=True,
        conf=0.5,
        iou=0.45,
        save_debug_visualizations=debug,
    )
    
    # Stage 3 Configuration
    segmentation_config = SegmentationConfig(
        model_type="mobilesam",
        checkpoint_path=Path("/app/models/mobilesam/mobile_sam.pt"),
        device="cuda:0",
        half=True,
        batch_size=8,
        save_overlays=debug,
        save_metadata=True,
        save_manifest=True,
    )
    
    # Find all images
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # ========================================
    # STAGE 2: FACE DETECTION
    # ========================================
    logger.info("=" * 80)
    logger.info("STAGE 2: FACE DETECTION")
    logger.info("=" * 80)
    
    # Load YOLO model
    yolo_model = load_detection_model(detection_config)
    
    detection_results = []
    for image_path in image_files:
        logger.info(f"\nProcessing: {image_path.name}")
        
        # Create metadata
        image_metadata = create_mock_image_metadata(image_path)
        
        # Run detection
        det_result = detect_faces_in_image(
            image_path,
            image_metadata,
            yolo_model,
            detection_config
        )
        
        # Save debug visualization
        if debug and det_result.debug_image is not None:
            save_detection_debug(det_result, output_dir)
        
        # Store result
        detection_results.append((image_path, det_result))
        
        logger.info(det_result.get_summary())
    
    # CRITICAL: Unload YOLO before loading SAM
    logger.info("\nUnloading YOLO model...")
    unload_detection_model(yolo_model)
    yolo_model = None
    
    logger.info("\n" + "=" * 80)
    logger.info("Detection stage complete!")
    logger.info(f"Total detections: {sum(r.detection_count for _, r in detection_results)}")
    logger.info("=" * 80)
    
    # ========================================
    # STAGE 3: FACE SEGMENTATION
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: FACE SEGMENTATION")
    logger.info("=" * 80)
    
    # Load SAM model (YOLO must be unloaded first!)
    sam_predictor = load_segmentation_model(segmentation_config)
    
    segmentation_results = []
    for image_path, det_result in detection_results:
        if det_result.detection_count == 0:
            logger.warning(f"Skipping {image_path.name} - no faces detected")
            continue
        
        logger.info(f"\nSegmenting faces in: {image_path.name}")
        
        # Run segmentation
        seg_result = segment_faces_in_image(
            image_path,
            det_result,
            sam_predictor,
            segmentation_config,
            output_dir
        )
        
        segmentation_results.append(seg_result)
        
        logger.info(seg_result.get_summary())
    
    # Unload SAM model
    logger.info("\nUnloading SAM model...")
    unload_segmentation_model(sam_predictor)
    sam_predictor = None
    
    logger.info("\n" + "=" * 80)
    logger.info("Segmentation stage complete!")
    logger.info(f"Total faces segmented: {sum(len(r.faces) for r in segmentation_results)}")
    logger.info("=" * 80)
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    total_images = len(image_files)
    total_detections = sum(r.detection_count for _, r in detection_results)
    total_segmented = sum(len(r.faces) for r in segmentation_results)
    total_failed = sum(r.failed_segmentations for r in segmentation_results)
    
    logger.info(f"Images Processed: {total_images}")
    logger.info(f"Faces Detected: {total_detections}")
    logger.info(f"Faces Segmented: {total_segmented}")
    logger.info(f"Failed Segmentations: {total_failed}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    # Print output structure
    logger.info("\nOutput structure:")
    logger.info(f"  Masks: {output_dir / 'masks'}")
    logger.info(f"  Metadata: {output_dir / 'metadata'}")
    if debug:
        logger.info(f"  Debug: {output_dir / 'debug'}")
    
    logger.success("\nPipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run face detection and segmentation pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data/preprocessed"),
        help="Directory containing preprocessed images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory for output files"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug visualizations"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            debug=not args.no_debug
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
