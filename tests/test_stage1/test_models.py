"""
Tests for data models.

Verifies the correctness of data structures used throughout the pipeline.
"""

import pytest
from datetime import datetime
from pathlib import Path

from cv_pipeline.stage1_ingestion.models import (
    ImageMetadata,
    ValidationResult,
    IngestionResult,
    ImageStatus,
    FailureReason
)


class TestImageMetadata:
    """Test ImageMetadata data structure."""
    
    def test_create_image_metadata(self):
        """Test creating ImageMetadata with required fields."""
        metadata = ImageMetadata(
            original_filename="20260315_1.jpg",
            original_path=Path("/test/20260315_1.jpg"),
            file_size_bytes=1024000,
            capture_date=datetime(2026, 3, 15),
            session_number=1,
            original_width=4032,
            original_height=3024,
            original_aspect_ratio=1.333
        )
        
        assert metadata.original_filename == "20260315_1.jpg"
        assert metadata.session_number == 1
        assert metadata.status == ImageStatus.PENDING
        assert len(metadata.preprocessing_operations) == 0
    
    def test_add_preprocessing_operation(self):
        """Test adding preprocessing operations to metadata."""
        metadata = ImageMetadata(
            original_filename="test.jpg",
            original_path=Path("/test/test.jpg"),
            file_size_bytes=1024000,
            capture_date=datetime(2026, 3, 15),
            session_number=1,
            original_width=1920,
            original_height=1080,
            original_aspect_ratio=1.778
        )
        
        metadata.add_preprocessing_op("rgb_conversion")
        metadata.add_preprocessing_op("bilateral_filter")
        
        assert len(metadata.preprocessing_operations) == 2
        assert "rgb_conversion" in metadata.preprocessing_operations
        assert "bilateral_filter" in metadata.preprocessing_operations
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary for serialization."""
        metadata = ImageMetadata(
            original_filename="test.jpg",
            original_path=Path("/test/test.jpg"),
            file_size_bytes=1024000,
            capture_date=datetime(2026, 3, 15),
            session_number=1,
            original_width=1920,
            original_height=1080,
            original_aspect_ratio=1.778
        )
        
        data = metadata.to_dict()
        
        assert isinstance(data, dict)
        assert data['original_filename'] == "test.jpg"
        assert data['session_number'] == 1
        assert isinstance(data['capture_date'], str)
        assert isinstance(data['status'], str)


class TestValidationResult:
    """Test ValidationResult data structure."""
    
    def test_create_validation_result(self):
        """Test creating ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            filename="20260315_1.jpg",
            file_path=Path("/test/20260315_1.jpg")
        )
        
        assert result.is_valid is True
        assert result.filename == "20260315_1.jpg"
        assert result.failure_reason is None
    
    def test_validation_failure_with_reason(self):
        """Test validation result with failure reason."""
        result = ValidationResult(
            is_valid=False,
            filename="invalid.jpg",
            file_path=Path("/test/invalid.jpg"),
            failure_reason=FailureReason.DIMENSIONS_TOO_SMALL,
            error_message="Image too small"
        )
        
        assert result.is_valid is False
        assert result.failure_reason == FailureReason.DIMENSIONS_TOO_SMALL
        assert result.error_message == "Image too small"


class TestIngestionResult:
    """Test IngestionResult data structure."""
    
    def test_create_ingestion_result(self):
        """Test creating IngestionResult."""
        result = IngestionResult()
        
        assert result.total_images_found == 0
        assert result.total_images_processed == 0
        assert result.total_images_failed == 0
    
    def test_add_successful_image(self):
        """Test adding successful image to results."""
        result = IngestionResult()
        
        metadata = ImageMetadata(
            original_filename="test.jpg",
            original_path=Path("/test/test.jpg"),
            file_size_bytes=1024000,
            capture_date=datetime(2026, 3, 15),
            session_number=1,
            original_width=1920,
            original_height=1080,
            original_aspect_ratio=1.778
        )
        
        result.add_successful_image(metadata)
        
        assert result.total_images_processed == 1
        assert len(result.successful_images) == 1
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = IngestionResult()
        result.total_images_found = 10
        
        # Add 7 successful images
        for i in range(7):
            metadata = ImageMetadata(
                original_filename=f"test{i}.jpg",
                original_path=Path(f"/test/test{i}.jpg"),
                file_size_bytes=1024000,
                capture_date=datetime(2026, 3, 15),
                session_number=1,
                original_width=1920,
                original_height=1080,
                original_aspect_ratio=1.778
            )
            result.add_successful_image(metadata)
        
        assert result.get_success_rate() == 70.0
    
    def test_get_summary(self):
        """Test summary string generation."""
        result = IngestionResult()
        result.total_images_found = 10
        result.total_images_processed = 8
        result.total_images_failed = 2
        
        summary = result.get_summary()
        
        assert "Total Found: 10" in summary
        assert "Processed: 8" in summary
        assert "Failed: 2" in summary
        assert "Success Rate: 80.00%" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
