"""
Data models for Stage 3: Face Segmentation

This module defines the core data structures used throughout the segmentation pipeline
to ensure type safety and strict data passing between components.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

from cv_pipeline.stage2_detection.models import Detection, DetectionResult
from cv_pipeline.stage1_ingestion.models import ImageMetadata


@dataclass
class SegmentationConfig:
    """
    Configuration for the face segmentation stage.
    
    Defines model parameters, prompting settings, and refinement options.
    """
    # Model configuration
    model_type: str = "mobilesam"
    checkpoint_path: Path = Path("/app/models/mobilesam/mobile_sam.pt")
    device: str = "cuda:0"
    half: bool = True  # FP16 inference
    
    # Inference settings
    multimask_output: bool = True  # Generate 3 masks, pick best
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    batch_size: int = 8  # Faces per batch
    
    # Prompting settings
    use_expanded_bbox: bool = True
    
    # Refinement settings
    refinement_enabled: bool = True
    open_kernel_size: int = 3
    close_kernel_size: int = 5
    fill_holes: bool = True
    keep_largest: bool = True
    smooth_edges: bool = True
    smooth_sigma: float = 1.5
    
    # Output settings
    output_format: str = "png"
    min_dimension: int = 112  # AdaFace requirement
    background_color: Tuple[int, int, int] = (0, 0, 0)  # RGB black
    save_metadata: bool = True
    save_manifest: bool = True
    crop_padding: int = 5  # Padding around mask crop
    
    # Debug settings
    save_overlays: bool = True
    save_individual_masks: bool = False
    
    # Memory settings
    aggressive_cleanup: bool = False  # Call empty_cache after each batch
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SegmentationConfig':
        """Create SegmentationConfig from dictionary."""
        model_cfg = config_dict.get('model', {})
        inference_cfg = config_dict.get('inference', {})
        refinement_cfg = config_dict.get('refinement', {})
        output_cfg = config_dict.get('output', {})
        debug_cfg = config_dict.get('debug', {})
        memory_cfg = config_dict.get('memory', {})
        
        return cls(
            model_type=model_cfg.get('type', 'mobilesam'),
            checkpoint_path=Path(model_cfg.get('checkpoint_path', '/app/models/mobilesam/mobile_sam.pt')),
            device=model_cfg.get('device', 'cuda:0'),
            half=model_cfg.get('half', True),
            multimask_output=inference_cfg.get('multimask_output', True),
            pred_iou_thresh=inference_cfg.get('pred_iou_thresh', 0.88),
            stability_score_thresh=inference_cfg.get('stability_score_thresh', 0.95),
            batch_size=inference_cfg.get('batch_size', 8),
            refinement_enabled=refinement_cfg.get('enabled', True),
            open_kernel_size=refinement_cfg.get('open_kernel_size', 3),
            close_kernel_size=refinement_cfg.get('close_kernel_size', 5),
            fill_holes=refinement_cfg.get('fill_holes', True),
            keep_largest=refinement_cfg.get('keep_largest', True),
            smooth_edges=refinement_cfg.get('smooth_edges', True),
            smooth_sigma=refinement_cfg.get('smooth_sigma', 1.5),
            output_format=output_cfg.get('format', 'png'),
            min_dimension=output_cfg.get('min_dimension', 112),
            background_color=tuple(output_cfg.get('background_color', [0, 0, 0])),
            save_metadata=output_cfg.get('save_metadata', True),
            save_manifest=output_cfg.get('save_manifest', True),
            save_overlays=debug_cfg.get('save_overlays', True),
            save_individual_masks=debug_cfg.get('save_individual_masks', False),
            aggressive_cleanup=memory_cfg.get('aggressive_cleanup', False),
        )


@dataclass
class SegmentationResult:
    """
    Result of segmenting a single face.
    
    Contains the mask, score, and metadata for one face detection.
    """
    # Core segmentation data
    mask: Optional[np.ndarray] = None  # Binary mask (H, W)
    score: float = 0.0  # SAM IoU score
    
    # Reference to original detection
    detection_id: str = ""
    bbox: Optional[np.ndarray] = None  # SAM prompt bbox
    original_bbox: Optional[np.ndarray] = None  # Original YOLO bbox
    original_confidence: float = 0.0  # YOLO confidence
    
    # Mask properties
    mask_area_pixels: int = 0
    
    # Crop metadata
    crop_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    crop_width: int = 0
    crop_height: int = 0
    was_upscaled: bool = False
    
    # Quality assessment
    occlusion_score: float = 0.0
    segmentation_quality: str = "unknown"  # good, fair, poor
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'detection_id': self.detection_id,
            'score': float(self.score),
            'bbox': self.bbox.tolist() if self.bbox is not None else None,
            'original_bbox': self.original_bbox.tolist() if self.original_bbox is not None else None,
            'original_confidence': float(self.original_confidence),
            'mask_area_pixels': self.mask_area_pixels,
            'crop_bbox': self.crop_bbox,
            'crop_width': self.crop_width,
            'crop_height': self.crop_height,
            'was_upscaled': self.was_upscaled,
            'occlusion_score': float(self.occlusion_score),
            'segmentation_quality': self.segmentation_quality,
            'error': self.error
        }


@dataclass
class FaceOutput:
    """
    Information about a saved face output file.
    """
    path: Path
    metadata_path: Path
    confidence: float
    sam_score: float
    detection_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'output_file': self.path.name,
            'metadata_file': self.metadata_path.name,
            'detection_id': self.detection_id,
            'confidence': float(self.confidence),
            'sam_score': float(self.sam_score)
        }


@dataclass
class SessionSegmentationResult:
    """
    Result of segmenting all faces in a single image/session.
    """
    # Source image metadata
    image_metadata: ImageMetadata
    
    # Segmentation results
    faces: List[FaceOutput] = field(default_factory=list)
    failed_segmentations: int = 0
    
    # Processing metrics
    processing_time_seconds: float = 0.0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_face(self, face: FaceOutput) -> None:
        """Add a successfully segmented face."""
        self.faces.append(face)
    
    def add_error(self, detection_id: str, error: str, details: str = "") -> None:
        """Add a failed segmentation."""
        self.failed_segmentations += 1
        self.errors.append({
            'detection_id': detection_id,
            'error': error,
            'details': details
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for manifest."""
        return {
            'session_id': f"{self.image_metadata.capture_date.strftime('%Y%m%d')}_session_{self.image_metadata.session_number}",
            'source_image': self.image_metadata.original_filename,
            'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_faces_detected': len(self.faces) + self.failed_segmentations,
            'total_faces_segmented': len(self.faces),
            'failed_segmentations': self.failed_segmentations,
            'processing_time_seconds': self.processing_time_seconds,
            'faces': [face.to_dict() for face in self.faces],
            'errors': self.errors
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Segmentation Summary for {self.image_metadata.original_filename}:\n"
            f"  Total Faces Segmented: {len(self.faces)}\n"
            f"  Failed: {self.failed_segmentations}\n"
            f"  Processing Time: {self.processing_time_seconds:.1f}s"
        )


@dataclass
class BatchSegmentationResult:
    """
    Results for a batch of images processed through segmentation stage.
    """
    results: List[SessionSegmentationResult] = field(default_factory=list)
    total_images: int = 0
    total_faces: int = 0
    total_failures: int = 0
    total_processing_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_result(self, result: SessionSegmentationResult) -> None:
        """Add a session result to the batch."""
        self.results.append(result)
        self.total_images += 1
        self.total_faces += len(result.faces)
        self.total_failures += result.failed_segmentations
        self.total_processing_time_seconds += result.processing_time_seconds
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        avg_time = self.total_processing_time_seconds / max(self.total_images, 1)
        avg_faces = self.total_faces / max(self.total_images, 1)
        
        return (
            f"Batch Segmentation Summary:\n"
            f"  Total Images: {self.total_images}\n"
            f"  Total Faces Segmented: {self.total_faces}\n"
            f"  Total Failures: {self.total_failures}\n"
            f"  Average Faces/Image: {avg_faces:.1f}\n"
            f"  Average Time/Image: {avg_time:.1f}s\n"
            f"  Total Time: {self.total_processing_time_seconds:.1f}s"
        )
