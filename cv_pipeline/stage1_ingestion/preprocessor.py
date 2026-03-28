"""
Image preprocessing module for Stage 1 ingestion.

Implements WhatsApp-focused preprocessing pipeline:
1. RGB color space conversion
2. Resolution normalization (FIRST for performance)
3. Bilateral filtering (JPEG artifact reduction)
4. CLAHE (lighting uniformization)

Performance target: 2-5 seconds per image
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import cv2
from PIL import Image
from loguru import logger

from .models import PreprocessingResult, FailureReason
from ..utils.exceptions import PreprocessingError


class ImagePreprocessor:
    """
    Preprocessor optimized for WhatsApp group photos.
    
    Handles severe JPEG compression artifacts and lighting variations
    while maintaining processing speed within 2-5 second budget.
    """
    
    def __init__(
        self,
        max_dimension: int = 2048,
        resize_interpolation: str = "INTER_AREA",
        ensure_rgb: bool = True,
        bilateral_enabled: bool = True,
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        output_quality: int = 95
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            max_dimension: Maximum dimension (width or height) for normalization
            resize_interpolation: OpenCV interpolation method for resizing
            ensure_rgb: Convert images to RGB color space
            bilateral_enabled: Enable bilateral filtering
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter sigma in color space
            bilateral_sigma_space: Bilateral filter sigma in coordinate space
            clahe_enabled: Enable CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid_size: CLAHE tile grid size
            output_quality: JPEG output quality (0-100)
        """
        self.max_dimension = max_dimension
        self.resize_interpolation = self._get_cv2_interpolation(resize_interpolation)
        self.ensure_rgb = ensure_rgb
        self.bilateral_enabled = bilateral_enabled
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_enabled = clahe_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.output_quality = output_quality
        
        # Create CLAHE object once for efficiency
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
    
    def _get_cv2_interpolation(self, method: str) -> int:
        """Convert interpolation method string to OpenCV constant."""
        methods = {
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4
        }
        return methods.get(method, cv2.INTER_AREA)
    
    def preprocess(
        self,
        input_path: Path,
        output_path: Path,
        save_intermediate: bool = False,
        intermediate_dir: Optional[Path] = None
    ) -> PreprocessingResult:
        """
        Execute complete preprocessing pipeline on an image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save preprocessed image
            save_intermediate: Whether to save intermediate processing steps
            intermediate_dir: Directory for intermediate outputs (if enabled)
        
        Returns:
            PreprocessingResult with operation details and metrics
        """
        start_time = time.time()
        operations_applied = []
        
        result = PreprocessingResult(
            success=False,
            input_path=input_path,
            output_path=output_path
        )
        
        try:
            # Load image using PIL first to ensure it's valid
            logger.debug(f"Loading image: {input_path.name}")
            pil_image = Image.open(input_path)
            
            # Convert PIL image to numpy array for OpenCV processing
            image = np.array(pil_image)
            original_height, original_width = image.shape[:2]
            result.original_dimensions = (original_width, original_height)
            
            logger.debug(
                f"Original dimensions: {original_width}x{original_height}, "
                f"channels: {image.shape[2] if len(image.shape) > 2 else 1}"
            )
            
            # STEP 1: Ensure RGB color space (CRITICAL for consistency)
            if self.ensure_rgb:
                image = self._convert_to_rgb(image, pil_image.mode)
                operations_applied.append("rgb_conversion")
                
                if save_intermediate and intermediate_dir:
                    self._save_intermediate(
                        image, intermediate_dir, input_path.stem, "01_rgb"
                    )
            
            # STEP 2: Resolution normalization (MUST BE FIRST for performance)
            # This dramatically reduces processing time for subsequent operations
            if self._needs_resizing(image):
                image = self._normalize_resolution(image)
                operations_applied.append("resolution_normalization")
                
                logger.debug(
                    f"Resized to: {image.shape[1]}x{image.shape[0]} "
                    f"(max_dim={self.max_dimension})"
                )
                
                if save_intermediate and intermediate_dir:
                    self._save_intermediate(
                        image, intermediate_dir, input_path.stem, "02_resized"
                    )
            
            # STEP 3: Bilateral filtering (WhatsApp JPEG artifact reduction)
            if self.bilateral_enabled:
                image = self._apply_bilateral_filter(image)
                operations_applied.append("bilateral_filter")
                
                logger.debug(
                    f"Applied bilateral filter: d={self.bilateral_d}, "
                    f"sigma_color={self.bilateral_sigma_color}, "
                    f"sigma_space={self.bilateral_sigma_space}"
                )
                
                if save_intermediate and intermediate_dir:
                    self._save_intermediate(
                        image, intermediate_dir, input_path.stem, "03_bilateral"
                    )
            
            # STEP 4: CLAHE (mat lighting uniformization)
            if self.clahe_enabled:
                image = self._apply_clahe(image)
                operations_applied.append("clahe")
                
                logger.debug(
                    f"Applied CLAHE: clip_limit={self.clahe_clip_limit}, "
                    f"tile_grid={self.clahe_tile_grid_size}"
                )
                
                if save_intermediate and intermediate_dir:
                    self._save_intermediate(
                        image, intermediate_dir, input_path.stem, "04_clahe"
                    )
            
            # Save final preprocessed image
            self._save_image(image, output_path)
            
            final_height, final_width = image.shape[:2]
            result.final_dimensions = (final_width, final_height)
            result.operations_applied = operations_applied
            result.output_path = output_path
            result.success = True
            
            processing_time = time.time() - start_time
            result.processing_time_seconds = processing_time
            
            logger.debug(
                f"Preprocessing complete for {input_path.name}: "
                f"{processing_time:.2f}s, operations: {', '.join(operations_applied)}"
            )
            
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.failure_reason = FailureReason.PREPROCESSING_ERROR
            result.processing_time_seconds = time.time() - start_time
            
            logger.error(f"Preprocessing failed for {input_path.name}: {e}")
            
            return result
    
    def _convert_to_rgb(self, image: np.ndarray, original_mode: str) -> np.ndarray:
        """
        Ensure image is in RGB color space.
        
        Args:
            image: Input image array
            original_mode: Original PIL image mode
        
        Returns:
            Image in RGB color space
        """
        # Handle different color spaces
        if original_mode == 'L' or len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif original_mode == 'RGBA' or image.shape[2] == 4:
            # RGBA to RGB (remove alpha channel)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif original_mode == 'BGR' or (len(image.shape) == 3 and image.shape[2] == 3):
            # PIL loads as RGB, but verify
            # OpenCV uses BGR, but since we loaded with PIL, it's already RGB
            pass
        
        return image
    
    def _needs_resizing(self, image: np.ndarray) -> bool:
        """Check if image needs to be resized based on max_dimension."""
        height, width = image.shape[:2]
        return max(height, width) > self.max_dimension
    
    def _normalize_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image resolution by limiting longest dimension.
        
        Maintains aspect ratio. Uses INTER_AREA for high-quality downsampling.
        
        Args:
            image: Input image array
        
        Returns:
            Resized image maintaining aspect ratio
        """
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        if height > width:
            new_height = self.max_dimension
            new_width = int(width * (self.max_dimension / height))
        else:
            new_width = self.max_dimension
            new_height = int(height * (self.max_dimension / width))
        
        # Resize using specified interpolation
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=self.resize_interpolation
        )
        
        return resized
    
    def _apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to reduce JPEG compression artifacts.
        
        Bilateral filtering smooths images while preserving edges,
        making it ideal for removing compression blocks from WhatsApp images.
        
        Args:
            image: Input image in RGB format
        
        Returns:
            Filtered image
        """
        # Apply bilateral filter
        # Note: cv2.bilateralFilter works on each channel independently
        filtered = cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        return filtered
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Applied to L channel of LAB color space to uniformize mat lighting
        without affecting color.
        
        Args:
            image: Input image in RGB format
        
        Returns:
            Image with CLAHE applied
        """
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        l_channel_clahe = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
        
        # Convert back to RGB
        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return image_clahe
    
    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save image using OpenCV with specified quality.
        
        Args:
            image: Image array in RGB format
            output_path: Path to save the image
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save with specified quality
        cv2.imwrite(
            str(output_path),
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.output_quality]
        )
    
    def _save_intermediate(
        self,
        image: np.ndarray,
        intermediate_dir: Path,
        stem: str,
        step_name: str
    ) -> None:
        """
        Save intermediate processing step for debugging.
        
        Args:
            image: Image array to save
            intermediate_dir: Directory for intermediate outputs
            stem: Original filename stem
            step_name: Name of processing step
        """
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / f"{stem}_{step_name}.jpg"
        self._save_image(image, output_path)


def create_preprocessor_from_config(config) -> ImagePreprocessor:
    """
    Create an ImagePreprocessor from pipeline configuration.
    
    Args:
        config: PipelineConfig object
    
    Returns:
        Configured ImagePreprocessor instance
    """
    return ImagePreprocessor(
        max_dimension=config.max_dimension,
        resize_interpolation=config.resize_interpolation,
        ensure_rgb=config.ensure_rgb,
        bilateral_enabled=config.bilateral_filter_enabled,
        bilateral_d=config.bilateral_d,
        bilateral_sigma_color=config.bilateral_sigma_color,
        bilateral_sigma_space=config.bilateral_sigma_space,
        clahe_enabled=config.clahe_enabled,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_grid_size=config.clahe_tile_grid_size,
        output_quality=config.output_quality
    )
