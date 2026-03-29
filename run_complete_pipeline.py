#!/usr/bin/env python3
"""
Complete Pipeline Script (Stages 1, 2, 3)
==========================================

Processes all images in data/images/ through all three stages:
1. Stage 1: Data Ingestion & Preprocessing
2. Stage 2: Face Detection (YOLOv8)
3. Stage 3: Face Segmentation (MobileSAM)

Outputs:
- Preprocessed images: data/preprocessed/
- Detection results: output/debug/detections/
- Segmented faces: output/masks/{date}/session_{N}/
- Metadata: output/metadata/
"""

import sys
from pathlib import Path
import shutil
from typing import List
import time
import torch
from loguru import logger
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_IMAGES = project_root / "data" / "images"
DATA_RAW = project_root / "data" / "raw"
DATA_PREPROCESSED = project_root / "data" / "preprocessed"
OUTPUT_DIR = project_root / "output"

# ============================================================================
# STAGE 0: PREPARE IMAGES
# ============================================================================

def setup_logger():
    """Configure logger for console output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def convert_images_to_jpeg_format():
    """Convert JPEG images to proper JPEG format if needed."""
    logger.info("=" * 80)
    logger.info("STAGE 0: CONVERTING IMAGES TO PROPER JPEG FORMAT")
    logger.info("=" * 80)
    
    if not DATA_IMAGES.exists():
        logger.error(f"Images directory not found: {DATA_IMAGES}")
        return 0
    
    import cv2
    import os
    
    image_files = list(DATA_IMAGES.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpeg', '.jpg']]
    
    logger.info(f"Found {len(image_files)} images in {DATA_IMAGES}")
    
    # Convert all to data/raw with proper naming
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    for img_file in image_files:
        try:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read: {img_file.name}")
                continue
            
            # Keep the same filename (already has format YYYYMMDDN.jpeg)
            # Just change extension to .jpg if needed
            dest_name = img_file.stem + ".jpg"
            dest_path = DATA_RAW / dest_name
            
            # Save as JPEG
            cv2.imwrite(str(dest_path), img)
            logger.info(f"✓ Converted: {img_file.name} → {dest_name}")
            converted += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to convert {img_file.name}: {e}")
    
    logger.info(f"\nSuccessfully converted {converted}/{len(image_files)} images")
    return converted

# ============================================================================
# STAGE 1: INGESTION
# ============================================================================

def run_stage1():
    """Run Stage 1: Data Ingestion & Preprocessing."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: DATA INGESTION & PREPROCESSING")
    logger.info("=" * 80)
    
    try:
        # Run using the existing ingestion module
        cmd = [
            sys.executable, "-m",
            "cv_pipeline.stage1_ingestion.ingestion"
        ]
        
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.success("Stage 1 completed successfully")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"Stage 1 failed with return code {result.returncode}")
            if result.stderr:
                logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error running Stage 1: {e}")
        return False

# ============================================================================
# STAGE 2: DETECTION
# ============================================================================

def run_stage2():
    """Run Stage 2: Face Detection with YOLOv8."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: FACE DETECTION (YOLOv8)")
    logger.info("=" * 80)
    
    try:
        # Import detection modules
        from cv_pipeline.stage2_detection.detector import (
            load_detection_model,
            unload_detection_model,
            process_image as detect_image,
            save_detection_debug,  # FIXED: Added import (Issue #3)
        )
        from cv_pipeline.stage2_detection.models import DetectionConfig
        from cv_pipeline.stage1_ingestion.models import ImageMetadata, ImageStatus
        from datetime import datetime
        import cv2
        import gc
        
        MODEL_YOLO_PATH = project_root / "models" / "yolov8" / "yolov8n-face.pt"
        
        if not MODEL_YOLO_PATH.exists():
            logger.error(f"YOLOv8 model not found: {MODEL_YOLO_PATH}")
            return False
        
        # Configure detection (use FP32 to avoid dtype mismatch)
        config = DetectionConfig(
            weights_path=MODEL_YOLO_PATH,
            conf=0.3,
            iou=0.45,
            max_det=100,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            min_confidence=0.3,
            min_face_width=30,
            min_face_height=30,
            half=False,  # FP32 instead of FP16 to avoid dtype issues
            verbose=False
        )
        
        logger.info(f"Device: {config.device}")
        logger.info("Loading YOLOv8 model...")
        model = load_detection_model(config)
        
        # Find preprocessed images (Stage 1 outputs .jpeg)
        preprocessed_images = sorted(DATA_PREPROCESSED.glob("*.jpeg"))
        if not preprocessed_images:
            preprocessed_images = sorted(DATA_PREPROCESSED.glob("*.jpg"))
        
        if not preprocessed_images:
            logger.warning(f"No preprocessed images found in {DATA_PREPROCESSED}")
            logger.info("Skipping Stage 2")
            return True
        
        logger.info(f"Found {len(preprocessed_images)} preprocessed images")
        
        detection_results = {}
        successful = 0
        total_faces = 0
        
        for idx, img_path in enumerate(preprocessed_images, 1):
            try:
                logger.info(f"[{idx}/{len(preprocessed_images)}] Processing: {img_path.name}")
                
                # Create metadata
                stem = img_path.stem
                if len(stem) >= 8:
                    year = int(stem[:4])
                    month = int(stem[4:6])
                    day = int(stem[6:8])
                    session = int(stem[8:]) if len(stem) > 8 else 1
                else:
                    year, month, day, session = 2026, 1, 1, 1
                
                capture_date = datetime(year, month, day)
                
                # Read image dimensions
                img = cv2.imread(str(img_path))
                height, width = img.shape[:2] if img is not None else (0, 0)
                
                metadata = ImageMetadata(
                    original_filename=img_path.name,
                    original_path=img_path,
                    file_size_bytes=img_path.stat().st_size,
                    capture_date=capture_date,
                    session_number=session,
                    original_width=width,
                    original_height=height,
                    original_aspect_ratio=width / height if height > 0 else 0,
                    status=ImageStatus.VALID
                )
                
                # Detect faces
                result = detect_image(
                    image_path=img_path,
                    image_metadata=metadata,
                    model=model,
                    config=config
                )
                
                # FIXED: Save debug visualizations if enabled (Issue #3)
                if config.save_debug_visualizations and result.debug_image is not None:
                    save_detection_debug(result, OUTPUT_DIR)
                
                detection_results[img_path] = result
                faces_detected = len(result.detections)
                total_faces += faces_detected
                successful += 1
                
                logger.success(f"  ✓ Detected {faces_detected} faces")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
        
        # Unload model
        logger.info("Unloading YOLOv8 model...")
        unload_detection_model(model)
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.success(f"\nStage 2 completed: {successful} images, {total_faces} total faces")
        
        return detection_results if successful > 0 else {}
        
    except Exception as e:
        logger.error(f"Error in Stage 2: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# STAGE 3: SEGMENTATION
# ============================================================================

def run_stage3(detection_results):
    """Run Stage 3: Face Segmentation with MobileSAM."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: FACE SEGMENTATION (MobileSAM)")
    logger.info("=" * 80)
    
    if not detection_results:
        logger.warning("No detection results to process. Skipping Stage 3.")
        return False
    
    try:
        from cv_pipeline.stage3_segmentation.segmenter import (
            load_segmentation_model,
            unload_segmentation_model,
            process_image as segment_image,
        )
        from cv_pipeline.stage3_segmentation.models import SegmentationConfig
        import gc
        
        MODEL_SAM_PATH = project_root / "models" / "mobilesam" / "mobile_sam.pt"
        
        if not MODEL_SAM_PATH.exists():
            logger.error(f"MobileSAM model not found: {MODEL_SAM_PATH}")
            return False
        
        # Configure segmentation (use FP32 to avoid dtype mismatch)
        config = SegmentationConfig(
            model_type="vit_t",  # MobileSAM model variant
            checkpoint_path=MODEL_SAM_PATH,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            half=False,  # FP32 instead of FP16 to avoid dtype issues
            min_dimension=112,  # AdaFace requirement
        )
        
        logger.info(f"Device: {config.device}")
        logger.info("Loading MobileSAM model...")
        predictor = load_segmentation_model(config)
        
        # Group by date
        from datetime import datetime
        sessions_by_date = {}
        
        for image_path, det_result in detection_results.items():
            if det_result is None or len(det_result.detections) == 0:
                continue
            
            stem = image_path.stem
            if len(stem) >= 8:
                date = stem[:8]  # YYYYMMDD
            else:
                date = "20260101"
            
            if date not in sessions_by_date:
                sessions_by_date[date] = []
            sessions_by_date[date].append((image_path, det_result))
        
        logger.info(f"Processing {len(sessions_by_date)} unique dates")
        
        total_segmented = 0
        successful_sessions = 0
        
        for session_idx, (date, session_data) in enumerate(sorted(sessions_by_date.items()), 1):
            try:
                logger.info(f"\n[Session {session_idx}] Date: {date}")
                
                # Create output directory
                output_base = OUTPUT_DIR / "masks" / date / f"session_{session_idx}"
                output_base.mkdir(parents=True, exist_ok=True)
                
                session_segmented = 0
                
                for image_path, det_result in session_data:
                    logger.info(f"  Processing {image_path.name} ({len(det_result.detections)} faces)")
                    
                    # Segment faces
                    seg_result = segment_image(
                        image_path=image_path,
                        det_result=det_result,
                        predictor=predictor,
                        config=config,
                        output_dir=output_base
                    )
                    
                    session_segmented += len(seg_result.faces)  # FIXED: Use len(faces) instead of success_count attribute
                
                total_segmented += session_segmented
                successful_sessions += 1
                logger.success(f"  ✓ Session complete: {session_segmented} faces segmented")
                
            except Exception as e:
                logger.error(f"  ✗ Session failed: {e}")
        
        # Unload model
        logger.info("Unloading MobileSAM model...")
        unload_segmentation_model(predictor)
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.success(f"\nStage 3 completed: {total_segmented} faces segmented")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Stage 3: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute all pipeline stages."""
    setup_logger()
    
    logger.info("=" * 80)
    logger.info("BJJU ADMIN PORTAL - COMPLETE PIPELINE (STAGES 1, 2, 3)")
    logger.info("=" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Input images: {DATA_IMAGES}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    overall_start = time.time()
    
    # Stage 0: Prepare images
    images_converted = convert_images_to_jpeg_format()
    if images_converted == 0:
        logger.error("No images were converted!")
        return
    
    # Stage 1: Ingestion
    if not run_stage1():
        logger.error("Stage 1 failed!")
        return
    
    # Stage 2: Detection
    detection_results = run_stage2()
    if detection_results is False:
        logger.error("Stage 2 failed!")
        return
    
    # Stage 3: Segmentation
    if detection_results:
        if not run_stage3(detection_results):
            logger.error("Stage 3 failed!")
            return
    
    # Final summary
    total_time = time.time() - overall_start
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total processing time: {total_time / 60:.1f} minutes")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Segmented faces: {OUTPUT_DIR / 'masks'}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
