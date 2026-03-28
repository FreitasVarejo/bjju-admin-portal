"""
Main ingestion orchestrator for Stage 1: Data Ingestion & Preprocessing.

Coordinates scanning, validation, and preprocessing of images with
resilient error handling and comprehensive metadata generation.
"""

import json
import time
from pathlib import Path
from typing import Optional
import yaml

from loguru import logger

from .models import (
    ImageMetadata,
    ImageStatus,
    IngestionResult,
    PipelineConfig
)
from .scanner import create_scanner_from_config
from .validator import create_validator_from_config
from .preprocessor import create_preprocessor_from_config
from .logger import get_logger


class IngestionPipeline:
    """
    Orchestrates the complete Stage 1 ingestion pipeline.
    
    Implements resilient processing where one image failure does not
    stop the entire pipeline, with comprehensive logging and metadata.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        
        # Configure logger
        pipeline_logger = get_logger()
        pipeline_logger.configure(
            log_path=config.logs_path,
            log_filename=config.log_filename,
            level=config.log_level,
            format_string=config.log_format,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            console_output=config.console_output,
            json_logs=config.json_logs
        )
        
        # Create component instances
        self.scanner = create_scanner_from_config(config)
        self.validator = create_validator_from_config(config)
        self.preprocessor = create_preprocessor_from_config(config)
        
        # Ensure output directories exist
        self._ensure_directories()
        
        logger.info("Ingestion pipeline initialized")
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.config.preprocessed_images_path,
            self.config.logs_path,
            self.config.failed_images_path
        ]
        
        if self.config.save_intermediate_steps:
            directories.append(self.config.intermediate_dir)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def run(self) -> IngestionResult:
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            IngestionResult with comprehensive processing statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Stage 1: Data Ingestion & Preprocessing Pipeline")
        logger.info("=" * 80)
        
        start_time = time.time()
        result = IngestionResult()
        
        # Store configuration snapshot for reproducibility
        result.config_snapshot = self._create_config_snapshot()
        
        try:
            # Step 1: Scan directory for images
            logger.info(f"Scanning directory: {self.config.raw_images_path}")
            image_files = self.scanner.scan_directory(self.config.raw_images_path)
            
            result.total_images_found = len(image_files)
            logger.info(f"Found {result.total_images_found} candidate images")
            
            if result.total_images_found == 0:
                logger.warning("No images found to process")
                return result
            
            # Apply debug limit if configured
            if self.config.max_images_to_process > 0:
                image_files = image_files[:self.config.max_images_to_process]
                logger.info(
                    f"Debug mode: limiting to {self.config.max_images_to_process} images"
                )
            
            # Step 2: Process each image
            logger.info("Starting image processing")
            for idx, image_path in enumerate(image_files, 1):
                logger.info(f"Processing [{idx}/{len(image_files)}]: {image_path.name}")
                
                self._process_single_image(image_path, result)
            
            # Step 3: Calculate final statistics
            result.total_processing_time_seconds = time.time() - start_time
            result.calculate_statistics()
            
            # Step 4: Log summary
            self._log_final_summary(result)
            
            # Step 5: Save metadata
            if self.config.metadata_enabled:
                self._save_batch_metadata(result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline failed with unexpected error: {e}")
            result.total_processing_time_seconds = time.time() - start_time
            raise
    
    def _process_single_image(
        self,
        image_path: Path,
        batch_result: IngestionResult
    ) -> None:
        """
        Process a single image through the complete pipeline.
        
        Implements resilient error handling - failures do not stop processing.
        
        Args:
            image_path: Path to the image file
            batch_result: Batch result object to update
        """
        image_start_time = time.time()
        
        try:
            # Step 1: Parse filename
            is_valid, capture_date, session_num, error_msg = (
                self.scanner.validate_and_parse_file(image_path)
            )
            
            if not is_valid:
                logger.warning(
                    f"Invalid filename format: {image_path.name} - {error_msg}"
                )
                batch_result.add_skipped_image(image_path.name)
                return
            
            # Step 2: Validate image
            validation_result = self.validator.validate(
                image_path,
                capture_date,
                session_num
            )
            
            if not validation_result.is_valid:
                logger.warning(
                    f"Validation failed: {image_path.name} - "
                    f"{validation_result.failure_reason.value if validation_result.failure_reason else 'unknown'}"
                )
                batch_result.add_failed_image(validation_result)
                
                # Optionally copy failed image for debugging
                if self.config.save_failed_images:
                    self._save_failed_image(image_path)
                
                return
            
            # Step 3: Preprocess image
            output_filename = self._generate_output_filename(image_path)
            output_path = self.config.preprocessed_images_path / output_filename
            
            preprocess_result = self.preprocessor.preprocess(
                input_path=image_path,
                output_path=output_path,
                save_intermediate=self.config.save_intermediate_steps,
                intermediate_dir=self.config.intermediate_dir if self.config.save_intermediate_steps else None
            )
            
            if not preprocess_result.success:
                logger.error(
                    f"Preprocessing failed: {image_path.name} - "
                    f"{preprocess_result.error_message}"
                )
                validation_result.is_valid = False
                validation_result.failure_reason = preprocess_result.failure_reason
                validation_result.error_message = preprocess_result.error_message
                batch_result.add_failed_image(validation_result)
                return
            
            # Step 4: Create metadata
            metadata = ImageMetadata(
                original_filename=image_path.name,
                original_path=image_path,
                file_size_bytes=image_path.stat().st_size,
                capture_date=capture_date,
                session_number=session_num,
                original_width=validation_result.width,
                original_height=validation_result.height,
                original_aspect_ratio=validation_result.aspect_ratio,
                processed_path=output_path,
                processed_width=preprocess_result.final_dimensions[0],
                processed_height=preprocess_result.final_dimensions[1],
                status=ImageStatus.PREPROCESSED,
                preprocessing_operations=preprocess_result.operations_applied
            )
            
            # Add preprocessing details to metadata
            metadata.metadata['preprocessing'] = {
                'processing_time_seconds': preprocess_result.processing_time_seconds,
                'operations': preprocess_result.operations_applied,
                'original_dimensions': preprocess_result.original_dimensions,
                'final_dimensions': preprocess_result.final_dimensions
            }
            
            # Add validation details if configured
            if self.config.include_validation_results:
                metadata.metadata['validation'] = {
                    'file_size_mb': validation_result.file_size_mb,
                    'aspect_ratio': validation_result.aspect_ratio
                }
            
            # Step 5: Save individual metadata if configured
            if self.config.metadata_enabled:
                self._save_image_metadata(metadata)
            
            # Step 6: Record success
            processing_time = time.time() - image_start_time
            batch_result.add_successful_image(metadata)
            
            logger.success(
                f"Successfully processed {image_path.name} in {processing_time:.2f}s"
            )
            
            # Check if processing time exceeds threshold
            if processing_time > self.config.max_processing_time_per_image:
                logger.warning(
                    f"Performance warning: {image_path.name} took {processing_time:.2f}s "
                    f"(threshold: {self.config.max_processing_time_per_image}s)"
                )
            
        except Exception as e:
            logger.exception(f"Unexpected error processing {image_path.name}: {e}")
            
            if self.config.continue_on_error:
                batch_result.add_skipped_image(image_path.name)
            else:
                raise
    
    def _generate_output_filename(self, input_path: Path) -> str:
        """
        Generate output filename for preprocessed image.
        
        Maintains original filename for traceability.
        """
        return input_path.name
    
    def _save_failed_image(self, image_path: Path) -> None:
        """Copy failed image to failed directory for debugging."""
        try:
            import shutil
            dest_path = self.config.failed_images_path / image_path.name
            shutil.copy2(image_path, dest_path)
            logger.debug(f"Saved failed image to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to save failed image {image_path.name}: {e}")
    
    def _save_image_metadata(self, metadata: ImageMetadata) -> None:
        """Save metadata for individual image."""
        try:
            metadata_filename = self.config.metadata_filename.format(
                original_filename=metadata.original_path.stem
            )
            metadata_path = self.config.preprocessed_images_path / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.debug(f"Saved metadata to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.original_filename}: {e}")
    
    def _save_batch_metadata(self, result: IngestionResult) -> None:
        """Save batch processing metadata."""
        try:
            batch_metadata_path = self.config.preprocessed_images_path / "batch_metadata.json"
            
            with open(batch_metadata_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved batch metadata to: {batch_metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save batch metadata: {e}")
    
    def _create_config_snapshot(self) -> dict:
        """Create a snapshot of key configuration parameters."""
        return {
            'max_dimension': self.config.max_dimension,
            'bilateral_filter_enabled': self.config.bilateral_filter_enabled,
            'clahe_enabled': self.config.clahe_enabled,
            'min_dimensions': f"{self.config.min_width}x{self.config.min_height}",
            'aspect_ratio_range': f"{self.config.aspect_ratio_min}-{self.config.aspect_ratio_max}",
            'output_quality': self.config.output_quality
        }
    
    def _log_final_summary(self, result: IngestionResult) -> None:
        """Log final processing summary."""
        logger.info("=" * 80)
        logger.info("Pipeline Processing Complete")
        logger.info("=" * 80)
        logger.info(f"Total Images Found: {result.total_images_found}")
        logger.info(f"Successfully Processed: {result.total_images_processed}")
        logger.info(f"Failed: {result.total_images_failed}")
        logger.info(f"Skipped: {result.total_images_skipped}")
        logger.info(f"Success Rate: {result.get_success_rate():.2f}%")
        logger.info(f"Total Processing Time: {result.total_processing_time_seconds:.2f}s")
        logger.info(f"Average Time per Image: {result.average_processing_time_seconds:.2f}s")
        logger.info("=" * 80)


def load_config(config_path: Path) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to pipeline_config.yaml
    
    Returns:
        PipelineConfig object
    """
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = PipelineConfig.from_dict(config_dict)
    logger.info("Configuration loaded successfully")
    
    return config


def run_ingestion_pipeline(config_path: Path) -> IngestionResult:
    """
    Main entry point for running the ingestion pipeline.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        IngestionResult with processing statistics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create and run pipeline
    pipeline = IngestionPipeline(config)
    result = pipeline.run()
    
    return result


if __name__ == "__main__":
    """
    Command-line entry point for Stage 1 ingestion pipeline.
    """
    import sys
    
    # Default config path
    default_config = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_config
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        result = run_ingestion_pipeline(config_path)
        print("\n" + result.get_summary())
        sys.exit(0 if result.get_success_rate() > 0 else 1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
