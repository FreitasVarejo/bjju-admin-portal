"""
Structured logging configuration using loguru.

Provides resilient logging that documents failures, skipped files,
and preprocessing operations while ensuring one image failure doesn't stop the pipeline.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class PipelineLogger:
    """
    Centralized logger for the CV pipeline.
    
    Uses loguru for structured logging with automatic rotation, compression,
    and flexible formatting for both development and production environments.
    """
    
    def __init__(self):
        """Initialize the pipeline logger."""
        self._configured = False
    
    def configure(
        self,
        log_path: Path,
        log_filename: str = "pipeline_{time}.log",
        level: str = "INFO",
        format_string: Optional[str] = None,
        rotation: str = "100 MB",
        retention: str = "30 days",
        compression: str = "zip",
        console_output: bool = True,
        json_logs: bool = False
    ) -> None:
        """
        Configure the logger with specified parameters.
        
        Args:
            log_path: Directory to store log files
            log_filename: Log filename pattern (supports time formatting)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_string: Custom format string (uses default if None)
            rotation: When to rotate logs (size or time-based)
            retention: How long to keep old logs
            compression: Compression format for rotated logs
            console_output: Whether to also log to console
            json_logs: Whether to use JSON structured logging
        """
        if self._configured:
            logger.warning("Logger already configured, reconfiguring...")
            logger.remove()
        
        # Ensure log directory exists
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Default format for development
        if format_string is None:
            if json_logs:
                format_string = "{message}"
            else:
                format_string = (
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                )
        
        # Remove default logger
        logger.remove()
        
        # Add console handler if requested
        if console_output:
            logger.add(
                sys.stderr,
                format=format_string,
                level=level,
                colorize=True,
                serialize=json_logs
            )
        
        # Add file handler with rotation
        log_file_path = log_path / log_filename
        logger.add(
            str(log_file_path),
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=json_logs,
            enqueue=True,  # Thread-safe logging
            backtrace=True,  # Enable detailed error traces
            diagnose=True   # Enable variable values in traces
        )
        
        self._configured = True
        logger.info(f"Pipeline logger configured: level={level}, log_path={log_path}")
    
    @staticmethod
    def log_image_processing_start(filename: str, file_path: Path) -> None:
        """Log the start of image processing."""
        logger.info(f"Processing image: {filename} from {file_path}")
    
    @staticmethod
    def log_image_processing_success(
        filename: str,
        processing_time: float,
        operations: list[str]
    ) -> None:
        """Log successful image processing."""
        ops_str = ", ".join(operations)
        logger.success(
            f"Successfully processed {filename} in {processing_time:.2f}s "
            f"(operations: {ops_str})"
        )
    
    @staticmethod
    def log_image_processing_failure(
        filename: str,
        reason: str,
        error_message: Optional[str] = None
    ) -> None:
        """Log image processing failure."""
        msg = f"Failed to process {filename}: {reason}"
        if error_message:
            msg += f" - {error_message}"
        logger.error(msg)
    
    @staticmethod
    def log_image_skipped(filename: str, reason: str) -> None:
        """Log skipped image."""
        logger.warning(f"Skipped image {filename}: {reason}")
    
    @staticmethod
    def log_validation_failure(
        filename: str,
        check_name: str,
        expected: str,
        actual: str
    ) -> None:
        """Log validation check failure."""
        logger.warning(
            f"Validation failed for {filename}: {check_name} "
            f"(expected: {expected}, actual: {actual})"
        )
    
    @staticmethod
    def log_preprocessing_operation(
        filename: str,
        operation: str,
        details: Optional[str] = None
    ) -> None:
        """Log preprocessing operation."""
        msg = f"Applied {operation} to {filename}"
        if details:
            msg += f": {details}"
        logger.debug(msg)
    
    @staticmethod
    def log_batch_start(total_images: int) -> None:
        """Log the start of batch processing."""
        logger.info(f"Starting batch processing of {total_images} images")
    
    @staticmethod
    def log_batch_complete(
        total_found: int,
        processed: int,
        failed: int,
        skipped: int,
        total_time: float
    ) -> None:
        """Log batch processing completion."""
        success_rate = (processed / total_found * 100) if total_found > 0 else 0
        logger.info(
            f"Batch processing complete: "
            f"{processed}/{total_found} processed ({success_rate:.1f}% success), "
            f"{failed} failed, {skipped} skipped, "
            f"total time: {total_time:.2f}s"
        )
    
    @staticmethod
    def log_performance_warning(filename: str, processing_time: float, threshold: float) -> None:
        """Log performance warning when processing exceeds threshold."""
        logger.warning(
            f"Performance warning: {filename} took {processing_time:.2f}s "
            f"(threshold: {threshold}s)"
        )
    
    @staticmethod
    def log_config_loaded(config_path: Path) -> None:
        """Log successful configuration loading."""
        logger.info(f"Configuration loaded from {config_path}")
    
    @staticmethod
    def log_directory_scan(directory: Path, pattern: str) -> None:
        """Log directory scanning."""
        logger.info(f"Scanning directory {directory} for pattern: {pattern}")
    
    @staticmethod
    def log_files_found(count: int, directory: Path) -> None:
        """Log number of files found."""
        logger.info(f"Found {count} candidate files in {directory}")


# Global logger instance
pipeline_logger = PipelineLogger()


def get_logger():
    """Get the global pipeline logger instance."""
    return pipeline_logger
