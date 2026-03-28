"""
Image validation module for Stage 1 ingestion.

Implements validation checks for JPEG integrity, dimensions, aspect ratio,
and file size without loading entire images into memory initially.
"""

import struct
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from PIL import Image
from loguru import logger

from .models import ValidationResult, FailureReason
from ..utils.exceptions import ValidationError


class ImageValidator:
    """
    Validator for checking image files meet pipeline requirements.
    
    Performs multiple validation checks including:
    - JPEG format and integrity
    - Minimum dimensions
    - Aspect ratio constraints
    - File size limits
    """
    
    def __init__(
        self,
        min_width: int,
        min_height: int,
        aspect_ratio_min: float,
        aspect_ratio_max: float,
        max_file_size_mb: float,
        verify_integrity: bool = True
    ):
        """
        Initialize the image validator.
        
        Args:
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            aspect_ratio_min: Minimum aspect ratio (width/height)
            aspect_ratio_max: Maximum aspect ratio (width/height)
            max_file_size_mb: Maximum file size in megabytes
            verify_integrity: Whether to verify JPEG integrity
        """
        self.min_width = min_width
        self.min_height = min_height
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.verify_integrity = verify_integrity
    
    def validate(
        self,
        file_path: Path,
        capture_date: Optional[datetime] = None,
        session_number: Optional[int] = None
    ) -> ValidationResult:
        """
        Perform complete validation of an image file.
        
        Args:
            file_path: Path to the image file
            capture_date: Parsed capture date from filename (optional)
            session_number: Parsed session number from filename (optional)
        
        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(
            is_valid=True,  # Will be set to False if any check fails
            filename=file_path.name,
            file_path=file_path,
            capture_date=capture_date,
            session_number=session_number
        )
        
        # Check 1: File exists and is readable
        if not self._check_file_exists(file_path, result):
            return result
        
        # Check 2: File size
        if not self._check_file_size(file_path, result):
            return result
        
        # Check 3: JPEG format
        if not self._check_jpeg_format(file_path, result):
            return result
        
        # Check 4: JPEG integrity (optional, more thorough)
        if self.verify_integrity:
            if not self._check_jpeg_integrity(file_path, result):
                return result
        
        # Check 5: Get dimensions and validate
        if not self._check_dimensions(file_path, result):
            return result
        
        # Check 6: Aspect ratio
        if not self._check_aspect_ratio(result):
            return result
        
        # All checks passed
        result.is_valid = True
        logger.debug(f"Validation passed for {file_path.name}")
        
        return result
    
    def _check_file_exists(self, file_path: Path, result: ValidationResult) -> bool:
        """Check if file exists and is readable."""
        if not file_path.exists():
            result.is_valid = False
            result.failure_reason = FailureReason.FILE_NOT_FOUND
            result.error_message = "File does not exist"
            logger.error(f"File not found: {file_path}")
            return False
        
        if not file_path.is_file():
            result.is_valid = False
            result.failure_reason = FailureReason.INVALID_FORMAT
            result.error_message = "Path is not a file"
            logger.error(f"Path is not a file: {file_path}")
            return False
        
        return True
    
    def _check_file_size(self, file_path: Path, result: ValidationResult) -> bool:
        """Check file size is within acceptable limits."""
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        result.file_size_mb = file_size_mb
        
        if file_size_bytes > self.max_file_size_bytes:
            result.is_valid = False
            result.file_size_valid = False
            result.failure_reason = FailureReason.FILE_TOO_LARGE
            result.error_message = (
                f"File size {file_size_mb:.2f}MB exceeds maximum "
                f"{self.max_file_size_bytes / (1024 * 1024):.2f}MB"
            )
            logger.warning(f"File too large: {file_path.name} ({file_size_mb:.2f}MB)")
            return False
        
        result.file_size_valid = True
        return True
    
    def _check_jpeg_format(self, file_path: Path, result: ValidationResult) -> bool:
        """
        Check if file is a valid JPEG by reading magic bytes.
        
        This is a fast check that doesn't require loading the entire image.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 2 bytes (JPEG magic number: FF D8)
                magic = f.read(2)
                
                if magic != b'\xff\xd8':
                    result.is_valid = False
                    result.format_valid = False
                    result.failure_reason = FailureReason.INVALID_FORMAT
                    result.error_message = "Not a valid JPEG file (invalid magic bytes)"
                    logger.warning(f"Invalid JPEG magic bytes: {file_path.name}")
                    return False
                
                # Check JPEG end marker (optional but good practice)
                f.seek(-2, 2)  # Seek to last 2 bytes
                end_marker = f.read(2)
                
                if end_marker != b'\xff\xd9':
                    logger.debug(
                        f"JPEG file {file_path.name} missing standard end marker "
                        "(may still be valid)"
                    )
            
            result.format_valid = True
            return True
            
        except Exception as e:
            result.is_valid = False
            result.format_valid = False
            result.failure_reason = FailureReason.CORRUPTED_FILE
            result.error_message = f"Error reading file: {str(e)}"
            logger.error(f"Error reading {file_path.name}: {e}")
            return False
    
    def _check_jpeg_integrity(self, file_path: Path, result: ValidationResult) -> bool:
        """
        Verify JPEG integrity by attempting to load with PIL.
        
        This is more thorough but slower than magic byte checking.
        """
        try:
            with Image.open(file_path) as img:
                # Verify the image by loading it
                img.verify()
            
            result.integrity_valid = True
            return True
            
        except Exception as e:
            result.is_valid = False
            result.integrity_valid = False
            result.failure_reason = FailureReason.CORRUPTED_FILE
            result.error_message = f"JPEG integrity check failed: {str(e)}"
            logger.error(f"Integrity check failed for {file_path.name}: {e}")
            return False
    
    def _check_dimensions(self, file_path: Path, result: ValidationResult) -> bool:
        """
        Check image dimensions meet minimum requirements.
        
        Uses PIL to get dimensions efficiently without loading full image data.
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
            result.width = width
            result.height = height
            
            if width < self.min_width or height < self.min_height:
                result.is_valid = False
                result.dimensions_valid = False
                result.failure_reason = FailureReason.DIMENSIONS_TOO_SMALL
                result.error_message = (
                    f"Dimensions {width}x{height} below minimum "
                    f"{self.min_width}x{self.min_height}"
                )
                logger.warning(
                    f"Image {file_path.name} too small: {width}x{height}"
                )
                return False
            
            result.dimensions_valid = True
            return True
            
        except Exception as e:
            result.is_valid = False
            result.dimensions_valid = False
            result.failure_reason = FailureReason.CORRUPTED_FILE
            result.error_message = f"Error reading dimensions: {str(e)}"
            logger.error(f"Error reading dimensions for {file_path.name}: {e}")
            return False
    
    def _check_aspect_ratio(self, result: ValidationResult) -> bool:
        """
        Check aspect ratio is within acceptable range.
        
        Args:
            result: ValidationResult with width and height already populated
        
        Returns:
            True if aspect ratio is valid, False otherwise
        """
        if result.width is None or result.height is None:
            result.is_valid = False
            result.aspect_ratio_valid = False
            result.failure_reason = FailureReason.INVALID_ASPECT_RATIO
            result.error_message = "Cannot calculate aspect ratio (missing dimensions)"
            return False
        
        aspect_ratio = result.width / result.height
        result.aspect_ratio = aspect_ratio
        
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
            result.is_valid = False
            result.aspect_ratio_valid = False
            result.failure_reason = FailureReason.INVALID_ASPECT_RATIO
            result.error_message = (
                f"Aspect ratio {aspect_ratio:.2f} outside range "
                f"[{self.aspect_ratio_min:.2f}, {self.aspect_ratio_max:.2f}]"
            )
            logger.warning(
                f"Invalid aspect ratio for {result.filename}: {aspect_ratio:.2f}"
            )
            return False
        
        result.aspect_ratio_valid = True
        return True
    
    def quick_validate(self, file_path: Path) -> bool:
        """
        Perform quick validation check (format and dimensions only).
        
        Useful for fast filtering before more expensive operations.
        
        Args:
            file_path: Path to the image file
        
        Returns:
            True if image passes quick validation, False otherwise
        """
        try:
            # Check JPEG magic bytes
            with open(file_path, 'rb') as f:
                if f.read(2) != b'\xff\xd8':
                    return False
            
            # Check dimensions
            with Image.open(file_path) as img:
                width, height = img.size
                
            if width < self.min_width or height < self.min_height:
                return False
            
            aspect_ratio = width / height
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                return False
            
            return True
            
        except Exception:
            return False


def create_validator_from_config(config) -> ImageValidator:
    """
    Create an ImageValidator from pipeline configuration.
    
    Args:
        config: PipelineConfig object
    
    Returns:
        Configured ImageValidator instance
    """
    return ImageValidator(
        min_width=config.min_width,
        min_height=config.min_height,
        aspect_ratio_min=config.aspect_ratio_min,
        aspect_ratio_max=config.aspect_ratio_max,
        max_file_size_mb=config.max_file_size_mb,
        verify_integrity=config.verify_integrity
    )
