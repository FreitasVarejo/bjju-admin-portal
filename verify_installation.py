#!/usr/bin/env python3
"""
Verification Script for Stage 2 & 3 Implementation

This script verifies that all components are properly installed and can be imported.
Run this before executing the full pipeline to catch any issues early.
"""

import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")

def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor}.{version.micro} < 3.10")
        return False

def check_imports():
    """Check all required imports"""
    checks = []
    
    # Core dependencies
    try:
        import numpy as np
        print_success(f"numpy {np.__version__}")
        checks.append(True)
    except ImportError as e:
        print_error(f"numpy import failed: {e}")
        checks.append(False)
    
    try:
        import cv2
        print_success(f"opencv-python {cv2.__version__}")
        checks.append(True)
    except ImportError as e:
        print_error(f"opencv-python import failed: {e}")
        checks.append(False)
    
    try:
        import torch
        print_success(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            print_success(f"  CUDA available: {torch.cuda.get_device_name(0)}")
            print_success(f"  CUDA version: {torch.version.cuda}")
        else:
            print_warning("  CUDA not available - GPU acceleration disabled")
        checks.append(True)
    except ImportError as e:
        print_error(f"torch import failed: {e}")
        checks.append(False)
    
    try:
        import torchvision
        print_success(f"torchvision {torchvision.__version__}")
        checks.append(True)
    except ImportError as e:
        print_error(f"torchvision import failed: {e}")
        checks.append(False)
    
    try:
        from ultralytics import YOLO
        import ultralytics
        print_success(f"ultralytics {ultralytics.__version__}")
        checks.append(True)
    except ImportError as e:
        print_error(f"ultralytics import failed: {e}")
        checks.append(False)
    
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        print_success("mobile_sam (installed)")
        checks.append(True)
    except ImportError as e:
        print_error(f"mobile_sam import failed: {e}")
        print_warning("  Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
        checks.append(False)
    
    try:
        from loguru import logger
        print_success("loguru (installed)")
        checks.append(True)
    except ImportError as e:
        print_error(f"loguru import failed: {e}")
        checks.append(False)
    
    return all(checks)

def check_pipeline_imports():
    """Check pipeline module imports"""
    checks = []
    
    try:
        from cv_pipeline.stage2_detection.models import Detection, DetectionConfig, DetectionResult
        print_success("stage2_detection.models")
        checks.append(True)
    except ImportError as e:
        print_error(f"stage2_detection.models import failed: {e}")
        checks.append(False)
    
    try:
        from cv_pipeline.stage2_detection.detector import (
            load_detection_model,
            unload_detection_model,
            detect_faces_auto,
        )
        print_success("stage2_detection.detector")
        checks.append(True)
    except ImportError as e:
        print_error(f"stage2_detection.detector import failed: {e}")
        checks.append(False)
    
    try:
        from cv_pipeline.stage3_segmentation.models import (
            SegmentationConfig,
            SegmentationResult,
            SessionSegmentationResult,
        )
        print_success("stage3_segmentation.models")
        checks.append(True)
    except ImportError as e:
        print_error(f"stage3_segmentation.models import failed: {e}")
        checks.append(False)
    
    try:
        from cv_pipeline.stage3_segmentation.segmenter import (
            load_segmentation_model,
            unload_segmentation_model,
            segment_all_faces,
        )
        print_success("stage3_segmentation.segmenter")
        checks.append(True)
    except ImportError as e:
        print_error(f"stage3_segmentation.segmenter import failed: {e}")
        checks.append(False)
    
    return all(checks)

def check_model_files():
    """Check if model weight files exist"""
    checks = []
    
    yolo_path = Path("models/yolov8/yolov8n-face.pt")
    if yolo_path.exists():
        size_mb = yolo_path.stat().st_size / 1024 / 1024
        print_success(f"YOLOv8-face weights found ({size_mb:.1f}MB)")
        checks.append(True)
    else:
        print_warning(f"YOLOv8-face weights not found at {yolo_path}")
        print_warning("  Download from: https://github.com/derronqi/yolov8-face")
        checks.append(False)
    
    sam_path = Path("models/mobilesam/mobile_sam.pt")
    if sam_path.exists():
        size_mb = sam_path.stat().st_size / 1024 / 1024
        print_success(f"MobileSAM weights found ({size_mb:.1f}MB)")
        checks.append(True)
    else:
        print_warning(f"MobileSAM weights not found at {sam_path}")
        print_warning("  Download from: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        checks.append(False)
    
    return all(checks)

def check_directories():
    """Check required directory structure"""
    dirs = [
        "cv_pipeline/stage2_detection",
        "cv_pipeline/stage3_segmentation",
        "examples",
    ]
    
    all_exist = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 80)
    print("BJJU CV Pipeline - Stage 2 & 3 Verification")
    print("=" * 80)
    print()
    
    # Python version
    print("1. Checking Python version...")
    python_ok = check_python_version()
    print()
    
    # Core dependencies
    print("2. Checking core dependencies...")
    imports_ok = check_imports()
    print()
    
    # Pipeline imports
    print("3. Checking pipeline modules...")
    pipeline_ok = check_pipeline_imports()
    print()
    
    # Model files
    print("4. Checking model files...")
    models_ok = check_model_files()
    print()
    
    # Directory structure
    print("5. Checking directory structure...")
    dirs_ok = check_directories()
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_checks = [python_ok, imports_ok, pipeline_ok, models_ok, dirs_ok]
    
    if all(all_checks):
        print_success("All checks passed! Ready to run the pipeline.")
        print()
        print("Next steps:")
        print("  1. Place test images in data/test_images/")
        print("  2. Run: python examples/run_detection_segmentation.py")
        return 0
    else:
        print_error("Some checks failed. Please fix the issues above.")
        print()
        if not imports_ok:
            print("To install dependencies:")
            print("  pip install -r requirements.txt")
        if not models_ok:
            print("To download models:")
            print("  See INSTALLATION_STAGES_2_3.md Step 5")
        return 1

if __name__ == "__main__":
    sys.exit(main())
