"""
Tests for filename parser and scanner.

Verifies regex pattern matching and file discovery logic.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from cv_pipeline.stage1_ingestion.scanner import FilenameParser, ImageScanner


class TestFilenameParser:
    """Test FilenameParser for extracting metadata from filenames."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with standard regex pattern."""
        pattern = r"^(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([1-9])\.jpe?g$"
        return FilenameParser(pattern, case_insensitive=True)
    
    def test_valid_filename_jpg(self, parser):
        """Test parsing valid filename with .jpg extension."""
        is_valid, date, session, error = parser.parse("20260315_1.jpg")
        
        assert is_valid is True
        assert date == datetime(2026, 3, 15)
        assert session == 1
        assert error is None
    
    def test_valid_filename_jpeg(self, parser):
        """Test parsing valid filename with .jpeg extension."""
        is_valid, date, session, error = parser.parse("20260315_1.jpeg")
        
        assert is_valid is True
        assert date == datetime(2026, 3, 15)
        assert session == 1
    
    def test_valid_filename_session_9(self, parser):
        """Test parsing filename with session 9."""
        is_valid, date, session, error = parser.parse("20261231_9.jpg")
        
        assert is_valid is True
        assert date == datetime(2026, 12, 31)
        assert session == 9
    
    def test_invalid_filename_wrong_format(self, parser):
        """Test parsing filename with wrong format."""
        is_valid, date, session, error = parser.parse("image_001.jpg")
        
        assert is_valid is False
        assert date is None
        assert session is None
        assert error is not None
    
    def test_invalid_date(self, parser):
        """Test parsing filename with invalid date."""
        is_valid, date, session, error = parser.parse("20260230_1.jpg")
        
        assert is_valid is False
        assert "Invalid date" in error
    
    def test_invalid_session_zero(self, parser):
        """Test parsing filename with session 0 (invalid)."""
        is_valid, date, session, error = parser.parse("20260315_0.jpg")
        
        assert is_valid is False
    
    def test_case_insensitive_extension(self, parser):
        """Test case-insensitive matching for extensions."""
        is_valid_lower, _, _, _ = parser.parse("20260315_1.jpg")
        is_valid_upper, _, _, _ = parser.parse("20260315_1.JPG")
        
        assert is_valid_lower is True
        assert is_valid_upper is True


class TestImageScanner:
    """Test ImageScanner for file discovery."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def scanner(self):
        """Create scanner with standard configuration."""
        pattern = r"^(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([1-9])\.jpe?g$"
        parser = FilenameParser(pattern, case_insensitive=True)
        return ImageScanner(
            extensions=['.jpg', '.jpeg'],
            filename_parser=parser,
            recursive=False
        )
    
    def test_scan_empty_directory(self, scanner, temp_dir):
        """Test scanning empty directory."""
        files = scanner.scan_directory(temp_dir)
        
        assert len(files) == 0
    
    def test_scan_with_valid_files(self, scanner, temp_dir):
        """Test scanning directory with valid files."""
        # Create test files
        valid_files = [
            "20260315_1.jpg",
            "20260315_2.jpg",
            "20260316_1.jpeg"
        ]
        
        for filename in valid_files:
            (temp_dir / filename).touch()
        
        files = scanner.scan_directory(temp_dir)
        
        assert len(files) == 3
        # Check files are sorted by name
        assert files[0].name == "20260315_1.jpg"
        assert files[2].name == "20260316_1.jpeg"
    
    def test_scan_with_mixed_files(self, scanner, temp_dir):
        """Test scanning directory with mixed valid and invalid files."""
        # Create test files
        all_files = [
            "20260315_1.jpg",      # Valid
            "invalid.jpg",          # Invalid filename
            "20260315_2.txt",       # Invalid extension
            "20260316_1.jpeg"       # Valid
        ]
        
        for filename in all_files:
            (temp_dir / filename).touch()
        
        files = scanner.scan_directory(temp_dir)
        
        # Should only find 2 valid files
        assert len(files) == 2
        assert all(f.suffix.lower() in ['.jpg', '.jpeg'] for f in files)
    
    def test_scan_nonexistent_directory(self, scanner):
        """Test scanning non-existent directory raises error."""
        nonexistent = Path("/nonexistent/directory")
        
        with pytest.raises(FileNotFoundError):
            scanner.scan_directory(nonexistent)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
