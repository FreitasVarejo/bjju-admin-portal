"""
Custom exceptions for the CV pipeline.

Provides specific exception types for better error handling and debugging.
"""


class PipelineException(Exception):
    """Base exception for all pipeline-related errors."""
    pass


class ConfigurationError(PipelineException):
    """Raised when there's an issue with pipeline configuration."""
    pass


class ValidationError(PipelineException):
    """Raised when image validation fails."""
    pass


class PreprocessingError(PipelineException):
    """Raised when image preprocessing fails."""
    pass


class FileParsingError(PipelineException):
    """Raised when filename parsing fails."""
    pass


class ImageLoadError(PipelineException):
    """Raised when image cannot be loaded or is corrupted."""
    pass


class TimeoutError(PipelineException):
    """Raised when processing exceeds maximum allowed time."""
    pass
