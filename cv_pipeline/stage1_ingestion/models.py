"""
Data models for Stage 1: Data Ingestion & Preprocessing

This module defines the core data structures used throughout the ingestion pipeline
to ensure type safety and strict data passing between components.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class ImageStatus(Enum):
    """Enumeration of possible image processing states."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    PREPROCESSED = "preprocessed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureReason(Enum):
    """Enumeration of possible failure reasons."""
    INVALID_FILENAME = "invalid_filename"
    FILE_NOT_FOUND = "file_not_found"
    INVALID_FORMAT = "invalid_format"
    CORRUPTED_FILE = "corrupted_file"
    DIMENSIONS_TOO_SMALL = "dimensions_too_small"
    INVALID_ASPECT_RATIO = "invalid_aspect_ratio"
    FILE_TOO_LARGE = "file_too_large"
    PREPROCESSING_ERROR = "preprocessing_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ImageMetadata:
    """
    Core metadata for an image in the pipeline.
    
    This structure carries essential information from ingestion through
    all pipeline stages, enabling traceability and future HITL integration.
    """
    # File information
    original_filename: str
    original_path: Path
    file_size_bytes: int
    
    # Parsed information from filename
    capture_date: datetime
    session_number: int
    
    # Image properties
    original_width: int
    original_height: int
    original_aspect_ratio: float
    
    # Processing information
    processed_path: Optional[Path] = None
    processed_width: Optional[int] = None
    processed_height: Optional[int] = None
    
    # Status tracking
    status: ImageStatus = ImageStatus.PENDING
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Preprocessing operations applied
    preprocessing_operations: List[str] = field(default_factory=list)
    
    # Additional metadata for downstream stages
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        data = asdict(self)
        # Convert Path objects to strings
        data['original_path'] = str(self.original_path)
        data['processed_path'] = str(self.processed_path) if self.processed_path else None
        # Convert datetime objects to ISO format
        data['capture_date'] = self.capture_date.isoformat()
        data['processing_timestamp'] = self.processing_timestamp.isoformat()
        # Convert enum to value
        data['status'] = self.status.value
        return data
    
    def add_preprocessing_op(self, operation: str) -> None:
        """Add a preprocessing operation to the history."""
        self.preprocessing_operations.append(operation)
    
    def update_processed_dimensions(self, width: int, height: int) -> None:
        """Update dimensions after preprocessing."""
        self.processed_width = width
        self.processed_height = height


@dataclass
class ValidationResult:
    """
    Result of image validation checks.
    
    Contains detailed information about validation status and any issues found.
    """
    is_valid: bool
    filename: str
    file_path: Path
    
    # Specific validation checks
    filename_valid: bool = False
    format_valid: bool = False
    integrity_valid: bool = False
    dimensions_valid: bool = False
    aspect_ratio_valid: bool = False
    file_size_valid: bool = False
    
    # Validation details
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    file_size_mb: Optional[float] = None
    
    # Error information
    failure_reason: Optional[FailureReason] = None
    error_message: Optional[str] = None
    
    # Parsed filename information (if valid)
    capture_date: Optional[datetime] = None
    session_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        data = asdict(self)
        data['file_path'] = str(self.file_path)
        data['failure_reason'] = self.failure_reason.value if self.failure_reason else None
        data['capture_date'] = self.capture_date.isoformat() if self.capture_date else None
        return data


@dataclass
class PreprocessingResult:
    """
    Result of image preprocessing operations.
    
    Tracks the success of preprocessing and provides detailed operation logs.
    """
    success: bool
    input_path: Path
    output_path: Optional[Path] = None
    
    # Processing metrics
    processing_time_seconds: float = 0.0
    operations_applied: List[str] = field(default_factory=list)
    
    # Dimension changes
    original_dimensions: tuple[int, int] = (0, 0)
    final_dimensions: tuple[int, int] = (0, 0)
    
    # Error information
    error_message: Optional[str] = None
    failure_reason: Optional[FailureReason] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preprocessing result to dictionary."""
        data = asdict(self)
        data['input_path'] = str(self.input_path)
        data['output_path'] = str(self.output_path) if self.output_path else None
        data['failure_reason'] = self.failure_reason.value if self.failure_reason else None
        return data


@dataclass
class IngestionResult:
    """
    Complete result of the ingestion pipeline for a batch of images.
    
    Provides comprehensive statistics and detailed results for monitoring
    and debugging the ingestion process.
    """
    # Overall statistics
    total_images_found: int = 0
    total_images_processed: int = 0
    total_images_failed: int = 0
    total_images_skipped: int = 0
    
    # Detailed results
    successful_images: List[ImageMetadata] = field(default_factory=list)
    failed_images: List[ValidationResult] = field(default_factory=list)
    skipped_images: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time_seconds: float = 0.0
    average_processing_time_seconds: float = 0.0
    
    # Timestamp
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration snapshot (for reproducibility)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def add_successful_image(self, metadata: ImageMetadata) -> None:
        """Add a successfully processed image to results."""
        self.successful_images.append(metadata)
        self.total_images_processed += 1
    
    def add_failed_image(self, validation_result: ValidationResult) -> None:
        """Add a failed image to results."""
        self.failed_images.append(validation_result)
        self.total_images_failed += 1
    
    def add_skipped_image(self, filename: str) -> None:
        """Add a skipped image to results."""
        self.skipped_images.append(filename)
        self.total_images_skipped += 1
    
    def calculate_statistics(self) -> None:
        """Calculate final statistics."""
        if self.total_images_processed > 0:
            self.average_processing_time_seconds = (
                self.total_processing_time_seconds / self.total_images_processed
            )
    
    def get_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_images_found == 0:
            return 0.0
        return (self.total_images_processed / self.total_images_found) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ingestion result to dictionary for serialization."""
        return {
            'total_images_found': self.total_images_found,
            'total_images_processed': self.total_images_processed,
            'total_images_failed': self.total_images_failed,
            'total_images_skipped': self.total_images_skipped,
            'success_rate_percent': self.get_success_rate(),
            'total_processing_time_seconds': self.total_processing_time_seconds,
            'average_processing_time_seconds': self.average_processing_time_seconds,
            'ingestion_timestamp': self.ingestion_timestamp.isoformat(),
            'successful_images': [img.to_dict() for img in self.successful_images],
            'failed_images': [img.to_dict() for img in self.failed_images],
            'skipped_images': self.skipped_images,
            'config_snapshot': self.config_snapshot
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of ingestion results."""
        return (
            f"Ingestion Summary:\n"
            f"  Total Found: {self.total_images_found}\n"
            f"  Processed: {self.total_images_processed}\n"
            f"  Failed: {self.total_images_failed}\n"
            f"  Skipped: {self.total_images_skipped}\n"
            f"  Success Rate: {self.get_success_rate():.2f}%\n"
            f"  Total Time: {self.total_processing_time_seconds:.2f}s\n"
            f"  Avg Time/Image: {self.average_processing_time_seconds:.2f}s"
        )


@dataclass
class PipelineConfig:
    """
    Configuration container for the ingestion pipeline.
    
    Provides type-safe access to configuration parameters loaded from YAML.
    """
    # Paths
    raw_images_path: Path
    preprocessed_images_path: Path
    logs_path: Path
    failed_images_path: Path
    
    # Parsing
    filename_pattern: str
    extensions: List[str]
    case_insensitive: bool
    recursive: bool
    
    # Validation
    min_width: int
    min_height: int
    aspect_ratio_min: float
    aspect_ratio_max: float
    max_file_size_mb: float
    verify_integrity: bool
    
    # Preprocessing
    max_dimension: int
    resize_interpolation: str
    ensure_rgb: bool
    bilateral_filter_enabled: bool
    bilateral_d: int
    bilateral_sigma_color: int
    bilateral_sigma_space: int
    clahe_enabled: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]
    output_format: str
    output_quality: int
    
    # Performance
    max_processing_time_per_image: int
    num_workers: int
    enable_profiling: bool
    
    # Logging
    log_level: str
    log_format: str
    json_logs: bool
    rotation: str
    retention: str
    compression: str
    log_filename: str
    console_output: bool
    
    # Metadata
    metadata_enabled: bool
    metadata_format: str
    include_preprocessing_ops: bool
    include_validation_results: bool
    metadata_filename: str
    
    # Error handling
    continue_on_error: bool
    save_failed_images: bool
    max_retries: int
    retry_delay: float
    
    # Debug
    save_intermediate_steps: bool
    intermediate_dir: Path
    visual_debug: bool
    max_images_to_process: int
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary loaded from YAML."""
        paths = config_dict['paths']
        parsing = config_dict['parsing']
        validation = config_dict['validation']
        preprocessing = config_dict['preprocessing']
        performance = config_dict['performance']
        logging_config = config_dict['logging']
        metadata = config_dict['metadata']
        error_handling = config_dict['error_handling']
        debug = config_dict['debug']
        
        return cls(
            # Paths
            raw_images_path=Path(paths['raw_images']),
            preprocessed_images_path=Path(paths['preprocessed_images']),
            logs_path=Path(paths['logs']),
            failed_images_path=Path(paths['failed_images']),
            
            # Parsing
            filename_pattern=parsing['filename_pattern'],
            extensions=parsing['extensions'],
            case_insensitive=parsing['case_insensitive'],
            recursive=parsing['recursive'],
            
            # Validation
            min_width=validation['min_width'],
            min_height=validation['min_height'],
            aspect_ratio_min=validation['aspect_ratio']['min'],
            aspect_ratio_max=validation['aspect_ratio']['max'],
            max_file_size_mb=validation['max_file_size_mb'],
            verify_integrity=validation['verify_integrity'],
            
            # Preprocessing
            max_dimension=preprocessing['max_dimension'],
            resize_interpolation=preprocessing['resize_interpolation'],
            ensure_rgb=preprocessing['ensure_rgb'],
            bilateral_filter_enabled=preprocessing['bilateral_filter']['enabled'],
            bilateral_d=preprocessing['bilateral_filter']['d'],
            bilateral_sigma_color=preprocessing['bilateral_filter']['sigma_color'],
            bilateral_sigma_space=preprocessing['bilateral_filter']['sigma_space'],
            clahe_enabled=preprocessing['clahe']['enabled'],
            clahe_clip_limit=preprocessing['clahe']['clip_limit'],
            clahe_tile_grid_size=tuple(preprocessing['clahe']['tile_grid_size']),
            output_format=preprocessing['output_format'],
            output_quality=preprocessing['output_quality'],
            
            # Performance
            max_processing_time_per_image=performance['max_processing_time_per_image'],
            num_workers=performance['num_workers'],
            enable_profiling=performance['enable_profiling'],
            
            # Logging
            log_level=logging_config['level'],
            log_format=logging_config['format'],
            json_logs=logging_config['json_logs'],
            rotation=logging_config['rotation'],
            retention=logging_config['retention'],
            compression=logging_config['compression'],
            log_filename=logging_config['log_filename'],
            console_output=logging_config['console_output'],
            
            # Metadata
            metadata_enabled=metadata['enabled'],
            metadata_format=metadata['format'],
            include_preprocessing_ops=metadata['include_preprocessing_ops'],
            include_validation_results=metadata['include_validation_results'],
            metadata_filename=metadata['metadata_filename'],
            
            # Error handling
            continue_on_error=error_handling['continue_on_error'],
            save_failed_images=error_handling['save_failed_images'],
            max_retries=error_handling['max_retries'],
            retry_delay=error_handling['retry_delay'],
            
            # Debug
            save_intermediate_steps=debug['save_intermediate_steps'],
            intermediate_dir=Path(debug['intermediate_dir']),
            visual_debug=debug['visual_debug'],
            max_images_to_process=debug['max_images_to_process']
        )
