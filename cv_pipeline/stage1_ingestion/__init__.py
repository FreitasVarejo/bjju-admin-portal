"""
Stage 1: Data Ingestion & Preprocessing

This module provides the complete pipeline for ingesting and preprocessing
images from WhatsApp, with focus on handling JPEG compression artifacts
and lighting variations.
"""

from .models import (
    ImageMetadata,
    ValidationResult,
    IngestionResult,
    PreprocessingResult,
    PipelineConfig,
    ImageStatus,
    FailureReason
)

from .scanner import ImageScanner, FilenameParser, create_scanner_from_config
from .validator import ImageValidator, create_validator_from_config
from .preprocessor import ImagePreprocessor, create_preprocessor_from_config
from .ingestion import IngestionPipeline, load_config, run_ingestion_pipeline
from .logger import get_logger

__all__ = [
    # Models
    'ImageMetadata',
    'ValidationResult',
    'IngestionResult',
    'PreprocessingResult',
    'PipelineConfig',
    'ImageStatus',
    'FailureReason',
    
    # Core classes
    'ImageScanner',
    'FilenameParser',
    'ImageValidator',
    'ImagePreprocessor',
    'IngestionPipeline',
    
    # Factory functions
    'create_scanner_from_config',
    'create_validator_from_config',
    'create_preprocessor_from_config',
    
    # Main functions
    'load_config',
    'run_ingestion_pipeline',
    'get_logger'
]
