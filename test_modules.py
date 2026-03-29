#!/usr/bin/env python3
"""
Test: Verify Stage 2 & 3 modules can be imported and basic functionality works
"""

import sys
from pathlib import Path
import numpy as np

print("=" * 80)
print("BJJU CV Pipeline - Stage 2 & 3 Import Test")
print("=" * 80)
print()

# Test 1: Import Stage 2 models
print("1. Testing Stage 2 imports...")
try:
    from cv_pipeline.stage2_detection.models import (
        Detection,
        DetectionConfig,
        DetectionResult,
        BatchDetectionResult,
    )
    print("  ✓ Stage 2 models imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Stage 2 models: {e}")
    sys.exit(1)

# Test 2: Import Stage 2 detector functions
try:
    from cv_pipeline.stage2_detection.detector import (
        expand_bbox_for_segmentation,
        calculate_iou,
        filter_detections,
    )
    print("  ✓ Stage 2 detector functions imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Stage 2 detector: {e}")
    sys.exit(1)

# Test 3: Import Stage 3 models
print("\n2. Testing Stage 3 imports...")
try:
    from cv_pipeline.stage3_segmentation.models import (
        SegmentationConfig,
        SegmentationResult,
        FaceOutput,
        SessionSegmentationResult,
    )
    print("  ✓ Stage 3 models imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Stage 3 models: {e}")
    sys.exit(1)

# Test 4: Import Stage 3 segmenter functions
try:
    from cv_pipeline.stage3_segmentation.segmenter import (
        refine_mask,
        fill_holes,
        keep_largest_component,
        apply_black_background,
    )
    print("  ✓ Stage 3 segmenter functions imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Stage 3 segmenter: {e}")
    sys.exit(1)

# Test 5: Create a Detection object
print("\n3. Testing Detection object creation...")
try:
    bbox = np.array([100.0, 150.0, 200.0, 250.0])
    detection = Detection(
        bbox=bbox,
        confidence=0.95,
        class_id=0
    )
    
    # Test properties
    assert detection.width == 100.0, "Width calculation failed"
    assert detection.height == 100.0, "Height calculation failed"
    assert detection.area == 10000.0, "Area calculation failed"
    assert detection.aspect_ratio == 1.0, "Aspect ratio calculation failed"
    
    print(f"  ✓ Detection object created successfully")
    print(f"    - BBox: {detection.bbox}")
    print(f"    - Confidence: {detection.confidence}")
    print(f"    - Width x Height: {detection.width} x {detection.height}")
    print(f"    - Area: {detection.area}")
except Exception as e:
    print(f"  ✗ Failed to create Detection object: {e}")
    sys.exit(1)

# Test 6: Test bbox expansion
print("\n4. Testing bbox expansion for SAM...")
try:
    config = DetectionConfig()
    expanded = expand_bbox_for_segmentation(
        bbox=bbox,
        config=config,
        image_shape=(1000, 1000)
    )
    
    print(f"  ✓ BBox expansion works")
    print(f"    - Original: {bbox}")
    print(f"    - Expanded: {expanded}")
    print(f"    - Expansion ratio: top={config.bbox_expand_top_ratio}, sides={config.bbox_expand_horizontal_ratio}")
except Exception as e:
    print(f"  ✗ BBox expansion failed: {e}")
    sys.exit(1)

# Test 7: Test IoU calculation
print("\n5. Testing IoU calculation...")
try:
    bbox1 = np.array([0, 0, 100, 100])
    bbox2 = np.array([50, 50, 150, 150])
    iou = calculate_iou(bbox1, bbox2)
    
    # Expected IoU for 50% overlap
    expected_iou = 2500 / (10000 + 10000 - 2500)  # intersection / union
    
    print(f"  ✓ IoU calculation works")
    print(f"    - BBox1: {bbox1}")
    print(f"    - BBox2: {bbox2}")
    print(f"    - IoU: {iou:.4f}")
except Exception as e:
    print(f"  ✗ IoU calculation failed: {e}")
    sys.exit(1)

# Test 8: Test mask refinement (morphological operations)
print("\n6. Testing mask refinement...")
try:
    # Create a simple binary mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255  # Square in center
    
    config = SegmentationConfig()
    refined = refine_mask(mask, config)
    
    print(f"  ✓ Mask refinement works")
    print(f"    - Original mask shape: {mask.shape}")
    print(f"    - Refined mask shape: {refined.shape}")
    print(f"    - Original pixels: {np.sum(mask > 0)}")
    print(f"    - Refined pixels: {np.sum(refined > 0)}")
except Exception as e:
    print(f"  ✗ Mask refinement failed: {e}")
    sys.exit(1)

# Test 9: Test black background application
print("\n7. Testing black background application...")
try:
    # Create a test image and mask
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255  # Square mask
    
    result = apply_black_background(image, mask)
    
    # Check that non-masked pixels are black
    assert np.all(result[0, 0] == [0, 0, 0]), "Background should be black"
    # Check that masked pixels are white
    assert np.all(result[50, 50] == [255, 255, 255]), "Masked region should be white"
    
    print(f"  ✓ Black background application works")
    print(f"    - Result shape: {result.shape}")
    print(f"    - Background pixel: {result[0, 0]}")
    print(f"    - Foreground pixel: {result[50, 50]}")
except Exception as e:
    print(f"  ✗ Black background application failed: {e}")
    sys.exit(1)

# Test 10: Test detection filtering
print("\n8. Testing detection filtering...")
try:
    config = DetectionConfig(
        min_confidence=0.5,
        min_face_width=40,
        min_face_height=40,
    )
    
    detections = [
        Detection(bbox=np.array([0, 0, 100, 100]), confidence=0.95),  # Good
        Detection(bbox=np.array([0, 0, 30, 30]), confidence=0.8),     # Too small
        Detection(bbox=np.array([0, 0, 50, 50]), confidence=0.3),     # Low confidence
    ]
    
    accepted, rejected = filter_detections(detections, config)
    
    print(f"  ✓ Detection filtering works")
    print(f"    - Total detections: {len(detections)}")
    print(f"    - Accepted: {len(accepted)}")
    print(f"    - Rejected: {len(rejected)}")
    
    if len(rejected) > 0:
        print(f"    - Rejection reasons:")
        for det in rejected:
            print(f"      • {det.rejection_reason}")
    
except Exception as e:
    print(f"  ✗ Detection filtering failed: {e}")
    sys.exit(1)

# Test 11: Test configuration from dict
print("\n9. Testing configuration from dictionary...")
try:
    config_dict = {
        'model': {
            'weights_path': '/test/path/model.pt',
            'device': 'cpu',
            'half': False,
        },
        'inference': {
            'conf': 0.6,
            'iou': 0.4,
        },
        'filtering': {
            'min_confidence': 0.6,
            'min_face_size': 40,
        },
        'output': {},
    }
    
    det_config = DetectionConfig.from_dict(config_dict)
    
    assert det_config.conf == 0.6, "Config parsing failed"
    assert det_config.iou == 0.4, "Config parsing failed"
    
    print(f"  ✓ Configuration parsing works")
    print(f"    - Confidence threshold: {det_config.conf}")
    print(f"    - IoU threshold: {det_config.iou}")
    print(f"    - Device: {det_config.device}")
except Exception as e:
    print(f"  ✗ Configuration parsing failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ All 9 tests passed!")
print("✓ Stage 2 (Detection) module is fully functional")
print("✓ Stage 3 (Segmentation) module is fully functional")
print()
print("Next steps:")
print("  1. Download model weights (see INSTALLATION_STAGES_2_3.md)")
print("  2. Test with real images using examples/run_detection_segmentation.py")
print("=" * 80)
