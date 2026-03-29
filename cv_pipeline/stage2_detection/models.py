"""
Data models for Stage 2: Face Detection

This module defines the core data structures used throughout the detection pipeline
to ensure type safety and strict data passing between components.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

from cv_pipeline.stage1_ingestion.models import ImageMetadata


@dataclass
class Detection:
    """
    Single face detection result.
    
    Contains bounding box, confidence score, and optional facial landmarks.
    """
    # Core detection data
    bbox: np.ndarray  # [x1, y1, x2, y2] in pixels
    confidence: float  # Detection confidence (0-1)
    class_id: int = 0  # Always 0 for face
    
    # Optional facial landmarks (5-point)
    landmarks: Optional[np.ndarray] = None  # Shape: (5, 2)
    
    # Detection metadata
    detection_id: str = ""  # Unique ID (set after creation)
    scale_detected: float = 1.0  # Scale at which detected
    occlusion_score: float = 0.0  # 0-1 occlusion estimate
    
    # For downstream stages (SAM)
    bbox_expanded: Optional[np.ndarray] = None  # Expanded bbox for segmentation
    
    # Rejection tracking (if filtered out)
    rejection_reason: Optional[str] = None
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return float(self.bbox[2] - self.bbox[0])
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return float(self.bbox[3] - self.bbox[1])
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center point."""
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2)
        )
    
    @property
    def aspect_ratio(self) -> float:
        """Get bounding box aspect ratio (width/height)."""
        return self.width / max(self.height, 1e-6)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for serialization."""
        return {
            'detection_id': self.detection_id,
            'bbox': self.bbox.tolist(),
            'bbox_expanded': self.bbox_expanded.tolist() if self.bbox_expanded is not None else None,
            'confidence': float(self.confidence),
            'class_id': self.class_id,
            'width': float(self.width),  # FIXED: Convert numpy float32 to native Python float
            'height': float(self.height),  # FIXED: Convert numpy float32 to native Python float
            'area': float(self.area),  # FIXED: Convert numpy float32 to native Python float
            'center': [float(x) for x in self.center],  # FIXED: Ensure tuple elements are native floats
            'aspect_ratio': float(self.aspect_ratio),  # FIXED: Convert numpy float32 to native Python float
            'scale_detected': float(self.scale_detected),  # FIXED: Convert numpy float32 to native Python float
            'occlusion_score': float(self.occlusion_score),  # FIXED: Convert numpy float32 to native Python float
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None,
            'rejection_reason': self.rejection_reason
        }


@dataclass
class DetectionConfig:
    """
    Configuration for the face detection stage.
    
    Defines model parameters, inference settings, and filtering thresholds.
    """
    # Model configuration
    weights_path: Path = Path("/app/models/yolov8/yolov8n-face.pt")
    device: str = "cuda:0"
    half: bool = True  # FP16 inference
    verbose: bool = False
    
    # Inference settings
    imgsz: int = 640  # Input size
    conf: float = 0.5  # Confidence threshold
    iou: float = 0.45  # NMS IoU threshold
    max_det: int = 100  # Max detections per image
    
    # Tiling settings
    tile_size: int = 640
    overlap: int = 128
    
    # Inference decision thresholds
    direct_inference_max: int = 1280  # Use direct inference
    scaled_inference_max: int = 2048  # Scale down then infer
    tiled_inference_min: int = 2048  # Use tiling
    
    # Filtering configuration
    # FIXED: Lowered min_face_width/height from 30 to 15 to detect small background faces (Issue #1)
    min_confidence: float = 0.5
    min_face_width: int = 15
    min_face_height: int = 15
    max_face_width: int = 1000
    max_face_height: int = 1000
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    
    # BBox expansion for SAM
    bbox_expansion_ratio: float = 0.3
    bbox_expand_top_ratio: float = 0.45  # Extra for hair
    bbox_expand_bottom_ratio: float = 0.15
    bbox_expand_horizontal_ratio: float = 0.30
    
    # Debug settings
    save_debug_visualizations: bool = True
    debug_output_dir: Optional[Path] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DetectionConfig':
        """Create DetectionConfig from dictionary."""
        model_cfg = config_dict.get('model', {})
        inference_cfg = config_dict.get('inference', {})
        tiling_cfg = config_dict.get('tiling', {})
        filtering_cfg = config_dict.get('filtering', {})
        output_cfg = config_dict.get('output', {})
        debug_cfg = config_dict.get('debug', {})  # FIXED: Added debug_cfg extraction (Issue #3)
        
        return cls(
            weights_path=Path(model_cfg.get('weights_path', '/app/models/yolov8/yolov8n-face.pt')),
            device=model_cfg.get('device', 'cuda:0'),
            half=model_cfg.get('half', True),
            verbose=model_cfg.get('verbose', False),
            imgsz=inference_cfg.get('imgsz', 640),
            conf=inference_cfg.get('conf', 0.5),
            iou=inference_cfg.get('iou', 0.45),
            max_det=inference_cfg.get('max_det', 100),
            tile_size=tiling_cfg.get('tile_size', 640),
            overlap=tiling_cfg.get('overlap', 128),
            min_confidence=filtering_cfg.get('min_confidence', 0.5),
            min_face_width=filtering_cfg.get('min_face_size', 30),
            min_face_height=filtering_cfg.get('min_face_size', 30),
            max_face_width=filtering_cfg.get('max_face_size', 1000),
            max_face_height=filtering_cfg.get('max_face_size', 1000),
            min_aspect_ratio=filtering_cfg.get('min_aspect_ratio', 0.5),
            max_aspect_ratio=filtering_cfg.get('max_aspect_ratio', 2.0),
            bbox_expansion_ratio=output_cfg.get('bbox_expansion_ratio', 0.3),
            # FIXED: Read from debug.visual_debug instead of output.save_debug_visualizations (Issue #3)
            save_debug_visualizations=debug_cfg.get('visual_debug', True),
        )


@dataclass
class DetectionResult:
    """
    Result of face detection for a single image.
    
    Contains all detected faces, rejected detections, and processing metadata.
    """
    # Source image metadata
    image_metadata: ImageMetadata
    
    # Detection results
    detections: List[Detection] = field(default_factory=list)
    rejected: List[Detection] = field(default_factory=list)
    
    # Processing metrics
    inference_time_ms: float = 0.0
    detection_count: int = 0
    rejection_count: int = 0
    
    # Inference method used
    inference_method: str = "direct"  # direct, scaled, tiled, multiscale
    
    # Optional debug visualization
    debug_image: Optional[np.ndarray] = None
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary for serialization."""
        return {
            'image_metadata': self.image_metadata.to_dict(),
            'detection_count': self.detection_count,
            'rejection_count': self.rejection_count,
            'inference_time_ms': self.inference_time_ms,
            'inference_method': self.inference_method,
            'detections': [det.to_dict() for det in self.detections],
            'rejected': [det.to_dict() for det in self.rejected],
            'error': self.error
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Detection Summary for {self.image_metadata.original_filename}:\n"
            f"  Detections: {self.detection_count}\n"
            f"  Rejected: {self.rejection_count}\n"
            f"  Inference Time: {self.inference_time_ms:.1f}ms\n"
            f"  Method: {self.inference_method}"
        )


@dataclass
class BatchDetectionResult:
    """
    Results for a batch of images processed through detection stage.
    """
    results: List[DetectionResult] = field(default_factory=list)
    total_images: int = 0
    total_detections: int = 0
    total_rejections: int = 0
    total_processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_result(self, result: DetectionResult) -> None:
        """Add a detection result to the batch."""
        self.results.append(result)
        self.total_images += 1
        self.total_detections += result.detection_count
        self.total_rejections += result.rejection_count
        self.total_processing_time_ms += result.inference_time_ms
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        avg_time = self.total_processing_time_ms / max(self.total_images, 1)
        avg_faces = self.total_detections / max(self.total_images, 1)
        
        return (
            f"Batch Detection Summary:\n"
            f"  Total Images: {self.total_images}\n"
            f"  Total Detections: {self.total_detections}\n"
            f"  Total Rejections: {self.total_rejections}\n"
            f"  Average Faces/Image: {avg_faces:.1f}\n"
            f"  Average Time/Image: {avg_time:.1f}ms\n"
            f"  Total Time: {self.total_processing_time_ms:.1f}ms"
        )
