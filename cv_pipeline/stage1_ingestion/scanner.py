"""
File scanner and parser for Stage 1 ingestion.

Implements non-recursive logic to search for JPEG files and extract
date and session information using the mandatory regex pattern.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from .models import FailureReason
from ..utils.exceptions import FileParsingError


class FilenameParser:
    """
    Parser for extracting metadata from BJJU image filenames.
    
    Filename format: YYYYMMDD{session}.jpg or YYYYMMDD{session}.jpeg
    Example: 20260315_1.jpg -> date: 2026-03-15, session: 1
    """
    
    def __init__(self, pattern: str, case_insensitive: bool = True):
        """
        Initialize the filename parser.
        
        Args:
            pattern: Regex pattern for filename validation and extraction
            case_insensitive: Whether to perform case-insensitive matching
        """
        flags = re.IGNORECASE if case_insensitive else 0
        self.pattern = re.compile(pattern, flags)
        self.case_insensitive = case_insensitive
    
    def parse(self, filename: str) -> Tuple[bool, Optional[datetime], Optional[int], Optional[str]]:
        """
        Parse filename to extract date and session information.
        
        Args:
            filename: The filename to parse
        
        Returns:
            Tuple of (is_valid, capture_date, session_number, error_message)
        """
        # Normalize filename for case-insensitive matching
        search_name = filename.lower() if self.case_insensitive else filename
        
        match = self.pattern.match(search_name)
        
        if not match:
            return False, None, None, f"Filename does not match pattern: {filename}"
        
        try:
            # Extract date components
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            session = int(match.group(4))
            
            # Validate and create date
            try:
                capture_date = datetime(year, month, day)
            except ValueError as e:
                return False, None, None, f"Invalid date in filename {filename}: {e}"
            
            # Validate session number
            if session < 1 or session > 9:
                return False, None, None, f"Invalid session number in {filename}: {session}"
            
            return True, capture_date, session, None
            
        except (IndexError, ValueError) as e:
            return False, None, None, f"Failed to parse filename {filename}: {e}"
    
    def is_valid_filename(self, filename: str) -> bool:
        """
        Check if filename matches the expected pattern.
        
        Args:
            filename: The filename to check
        
        Returns:
            True if filename is valid, False otherwise
        """
        is_valid, _, _, _ = self.parse(filename)
        return is_valid


class ImageScanner:
    """
    Scanner for discovering JPEG images in the input directory.
    
    Implements non-recursive scanning with extension filtering
    and filename validation.
    """
    
    def __init__(
        self,
        extensions: List[str],
        filename_parser: FilenameParser,
        recursive: bool = False
    ):
        """
        Initialize the image scanner.
        
        Args:
            extensions: List of file extensions to search for (e.g., ['.jpg', '.jpeg'])
            filename_parser: Parser for validating and extracting filename metadata
            recursive: Whether to search subdirectories (default: False for performance)
        """
        self.extensions = [ext.lower() for ext in extensions]
        self.filename_parser = filename_parser
        self.recursive = recursive
    
    def scan_directory(self, directory: Path) -> List[Path]:
        """
        Scan directory for image files matching the criteria.
        
        Args:
            directory: Directory to scan for images
        
        Returns:
            List of Path objects for valid image files
        
        Raises:
            FileNotFoundError: If directory does not exist
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        logger.info(f"Scanning directory: {directory} (recursive={self.recursive})")
        
        # Get all files matching extensions
        candidate_files = self._find_files_by_extension(directory)
        
        logger.info(f"Found {len(candidate_files)} files with valid extensions")
        
        # Filter by filename pattern
        valid_files = []
        invalid_count = 0
        
        for file_path in candidate_files:
            if self.filename_parser.is_valid_filename(file_path.name):
                valid_files.append(file_path)
            else:
                invalid_count += 1
                logger.debug(f"Skipping file with invalid filename: {file_path.name}")
        
        logger.info(
            f"Filename validation complete: {len(valid_files)} valid, "
            f"{invalid_count} invalid"
        )
        
        # Sort by filename (chronological order due to date-based naming)
        valid_files.sort(key=lambda p: p.name)
        
        return valid_files
    
    def _find_files_by_extension(self, directory: Path) -> List[Path]:
        """
        Find all files with specified extensions.
        
        Args:
            directory: Directory to search
        
        Returns:
            List of file paths with matching extensions
        """
        files = []
        
        if self.recursive:
            # Recursive search (not recommended for performance)
            for ext in self.extensions:
                files.extend(directory.rglob(f"*{ext}"))
                files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            # Non-recursive search (performance optimized)
            for item in directory.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in self.extensions:
                        files.append(item)
        
        return files
    
    def validate_and_parse_file(
        self,
        file_path: Path
    ) -> Tuple[bool, Optional[datetime], Optional[int], Optional[str]]:
        """
        Validate and parse a single file.
        
        Args:
            file_path: Path to the file to validate and parse
        
        Returns:
            Tuple of (is_valid, capture_date, session_number, error_message)
        """
        # Check file exists
        if not file_path.exists():
            return False, None, None, "File does not exist"
        
        # Check it's a file
        if not file_path.is_file():
            return False, None, None, "Path is not a file"
        
        # Parse filename
        is_valid, capture_date, session, error_msg = self.filename_parser.parse(
            file_path.name
        )
        
        return is_valid, capture_date, session, error_msg


def create_scanner_from_config(config) -> ImageScanner:
    """
    Create an ImageScanner from pipeline configuration.
    
    Args:
        config: PipelineConfig object
    
    Returns:
        Configured ImageScanner instance
    """
    parser = FilenameParser(
        pattern=config.filename_pattern,
        case_insensitive=config.case_insensitive
    )
    
    scanner = ImageScanner(
        extensions=config.extensions,
        filename_parser=parser,
        recursive=config.recursive
    )
    
    return scanner
