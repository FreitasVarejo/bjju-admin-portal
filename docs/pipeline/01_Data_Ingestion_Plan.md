# 01 - Data Ingestion Stage Plan

This document details the logic and implementation plan for the first stage of the CV pipeline: scanning input directories, parsing filenames, validating images, and preprocessing for downstream model consumption.

---

## 1. Stage Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA INGESTION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────┐    ┌────────────┐    ┌───────────┐    ┌────────────────────────┐ │
│  │   SCAN    │───▶│   PARSE    │───▶│  VALIDATE │───▶│     PREPROCESS        │ │
│  │ Directory │    │  Filename  │    │   Image   │    │  (Artifact Removal)   │ │
│  └───────────┘    └────────────┘    └───────────┘    └────────────────────────┘ │
│       │                │                  │                      │              │
│       ▼                ▼                  ▼                      ▼              │
│   List of         Metadata           Validated            Tensor-ready         │
│   .jpg files      (date,session)     Image objects        Images               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Scanning Logic

### 2.1 Scan Strategy

**Input Directory**: `./images/` (mounted as `/app/images/` in container)

**Scanning Approach**:
1. Non-recursive scan of the input directory (flat structure expected)
2. Filter for JPEG files only (`.jpg`, `.jpeg` extensions, case-insensitive)
3. Match against expected naming convention pattern
4. Collect non-matching files for warning logs

### 2.2 File Discovery Algorithm

```
FUNCTION scan_input_directory(input_path: Path) -> List[ImageFile]:
    
    all_files = list_files(input_path)
    valid_images = []
    invalid_files = []
    
    FOR each file IN all_files:
        IF file.extension.lower() IN ['.jpg', '.jpeg']:
            IF matches_naming_convention(file.name):
                valid_images.append(file)
            ELSE:
                invalid_files.append(file)
                LOG_WARNING(f"File {file.name} does not match naming convention")
        ELSE:
            invalid_files.append(file)
            LOG_DEBUG(f"Skipping non-JPEG file: {file.name}")
    
    LOG_INFO(f"Found {len(valid_images)} valid images, {len(invalid_files)} skipped")
    
    RETURN valid_images
```

### 2.3 Edge Cases

| Scenario | Handling |
|----------|----------|
| Empty directory | Log warning, return empty list, continue gracefully |
| No matching files | Log warning, return empty list |
| Mixed file types | Process only JPEG, log skipped files |
| Hidden files (`.xxx`) | Skip, do not log |
| Symlinks | Follow symlinks, validate target |
| Duplicate filenames | Should not occur (unique H value), log if detected |

---

## 3. Filename Parsing Logic

### 3.1 Naming Convention Specification

**Pattern**: `YYYYMMDDH.jpg`

| Component | Description | Example | Regex |
|-----------|-------------|---------|-------|
| `YYYY` | 4-digit year | 2023 | `[0-9]{4}` |
| `MM` | 2-digit month (01-12) | 10 | `(0[1-9]|1[0-2])` |
| `DD` | 2-digit day (01-31) | 27 | `(0[1-9]|[12][0-9]|3[01])` |
| `H` | Session order (1-9, single digit) | 1 | `[1-9]` |

**Full Regex Pattern**:
```regex
^(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([1-9])\.jpe?g$
```

**Note**: The single-digit `H` assumes a maximum of 9 sessions per day. If more sessions are expected, consider extending to `HH` (01-99).

### 3.2 Metadata Extraction Algorithm

```
FUNCTION parse_filename(filename: str) -> Optional[ImageMetadata]:
    
    pattern = r'^(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([1-9])\.jpe?g$'
    
    match = regex_match(pattern, filename, case_insensitive=True)
    
    IF match IS NULL:
        RETURN None
    
    year, month, day, session = match.groups()
    
    TRY:
        date = Date(year=int(year), month=int(month), day=int(day))
    EXCEPT InvalidDate:
        LOG_WARNING(f"Invalid date in filename: {filename}")
        RETURN None
    
    RETURN ImageMetadata(
        filename=filename,
        date=date,
        session=int(session),
        sort_key=f"{year}{month}{day}{session}"  # For chronological sorting
    )
```

### 3.3 ImageMetadata Data Structure

```
CLASS ImageMetadata:
    filename: str              # Original filename (e.g., "202310271.jpg")
    filepath: Path             # Full path to the file
    date: Date                 # Parsed date object
    session: int               # Session number (1-9)
    sort_key: str              # Sortable string (YYYYMMDDH)
    
    # Populated after validation
    width: Optional[int]
    height: Optional[int]
    file_size_bytes: Optional[int]
    
    # Populated after preprocessing
    preprocessed: bool = False
    preprocessing_applied: List[str] = []
```

---

## 4. Chronological Sorting

### 4.1 Sorting Strategy

**Primary Sort Key**: `sort_key` (YYYYMMDDH format)
**Sort Order**: Ascending (oldest first)

```
FUNCTION sort_images_chronologically(images: List[ImageMetadata]) -> List[ImageMetadata]:
    RETURN sorted(images, key=lambda img: img.sort_key)
```

### 4.2 Sorting Examples

| Filename | Sort Key | Order |
|----------|----------|-------|
| `202310271.jpg` | `202310271` | 1 |
| `202310272.jpg` | `202310272` | 2 |
| `202310281.jpg` | `202310281` | 3 |
| `202311011.jpg` | `202311011` | 4 |

### 4.3 Batch Processing Consideration

For the initial prototype, images will be processed sequentially in chronological order. Future optimizations may include:
- Parallel processing of images from different dates
- Priority queuing for recent dates
- Batch processing within memory constraints

---

## 5. Input Validation Plan

### 5.1 Validation Checks

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  File       │───▶│  Format     │───▶│  Resolution         │  │
│  │  Integrity  │    │  Validation │    │  Thresholds         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        │                  │                      │              │
│        ▼                  ▼                      ▼              │
│   Can file be        Is it valid          Min: 640x480         │
│   opened/read?       JPEG format?         Max: 8192x8192       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Validation Checks Detail

| Check | Criteria | Action on Fail |
|-------|----------|----------------|
| **File Exists** | File accessible at path | Log error, skip |
| **File Readable** | File can be opened | Log error, skip |
| **Valid JPEG** | JPEG magic bytes present | Log error, skip |
| **Not Truncated** | Complete image data | Log error, skip |
| **Min Resolution** | Width ≥ 640, Height ≥ 480 | Log warning, process with flag |
| **Max Resolution** | Width ≤ 8192, Height ≤ 8192 | Resize, log info |
| **Aspect Ratio** | Ratio between 0.25 and 4.0 | Log warning, process with flag |
| **Color Mode** | RGB or convertible | Convert to RGB, log info |

### 5.3 Validation Algorithm

```
FUNCTION validate_image(metadata: ImageMetadata) -> ValidationResult:
    
    result = ValidationResult(valid=True, warnings=[], errors=[])
    
    # Check 1: File exists and is readable
    IF NOT file_exists(metadata.filepath):
        result.errors.append("File not found")
        result.valid = False
        RETURN result
    
    # Check 2: Open and verify JPEG
    TRY:
        image = Image.open(metadata.filepath)
        image.verify()  # Verify without fully loading
    EXCEPT (IOError, SyntaxError) AS e:
        result.errors.append(f"Invalid or corrupted image: {e}")
        result.valid = False
        RETURN result
    
    # Re-open after verify (verify closes the file)
    image = Image.open(metadata.filepath)
    
    # Check 3: Resolution thresholds
    width, height = image.size
    metadata.width = width
    metadata.height = height
    
    IF width < 640 OR height < 480:
        result.warnings.append(f"Low resolution: {width}x{height}")
    
    IF width > 8192 OR height > 8192:
        result.warnings.append(f"Will be resized: {width}x{height} > 8192")
    
    # Check 4: Color mode
    IF image.mode NOT IN ['RGB', 'L']:
        IF image.mode IN ['RGBA', 'P', 'CMYK']:
            result.warnings.append(f"Color mode {image.mode} will be converted to RGB")
        ELSE:
            result.errors.append(f"Unsupported color mode: {image.mode}")
            result.valid = False
    
    # Check 5: Aspect ratio sanity
    aspect_ratio = width / height
    IF aspect_ratio < 0.25 OR aspect_ratio > 4.0:
        result.warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    
    # Store file size
    metadata.file_size_bytes = file_size(metadata.filepath)
    
    RETURN result
```

### 5.4 Validation Output Structure

```
CLASS ValidationResult:
    valid: bool                    # Overall validity
    errors: List[str]              # Blocking errors
    warnings: List[str]            # Non-blocking warnings
    metadata: ImageMetadata        # Updated metadata with dimensions
```

---

## 6. Preprocessing Plan (WhatsApp Artifact Handling)

### 6.1 WhatsApp Compression Characteristics

**Common Artifacts**:
- JPEG compression artifacts (blocky regions, mosquito noise)
- Resolution reduction (typically to ~1600x1200 or lower)
- Color space compression
- Loss of fine detail (hair, facial features)

**Impact on Pipeline**:
- Detection: Minimal impact (YOLOv8 robust to compression)
- Segmentation: Moderate impact (edge definition degradation)
- Downstream Recognition: Significant impact (fine feature loss)

### 6.2 Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │  Color      │───▶│  Denoising  │───▶│  CLAHE      │───▶│  Resolution     │   │
│  │  Correction │    │  (Light)    │    │  (Optional) │    │  Normalization  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────┘   │
│        │                  │                  │                    │             │
│        ▼                  ▼                  ▼                    ▼             │
│   RGB conversion    Reduce block        Enhance local       Cap at 2048px      │
│   if needed         artifacts           contrast                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Preprocessing Operations

#### 6.3.1 Color Correction

```
FUNCTION ensure_rgb(image: Image) -> Image:
    IF image.mode == 'RGB':
        RETURN image
    ELIF image.mode == 'RGBA':
        # Create white background and composite
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel
        RETURN background
    ELIF image.mode == 'L':
        RETURN image.convert('RGB')
    ELIF image.mode == 'P':
        RETURN image.convert('RGB')
    ELIF image.mode == 'CMYK':
        RETURN image.convert('RGB')
    ELSE:
        RAISE UnsupportedColorMode(image.mode)
```

#### 6.3.2 Light Denoising (JPEG Artifact Reduction)

**Strategy**: Apply bilateral filtering to reduce block artifacts while preserving edges.

**Parameters**:
- `d=5`: Diameter of pixel neighborhood
- `sigmaColor=75`: Filter sigma in color space
- `sigmaSpace=75`: Filter sigma in coordinate space

```
FUNCTION light_denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter for JPEG artifact reduction.
    Preserves edges while smoothing block artifacts.
    """
    # Bilateral filter: edge-preserving smoothing
    denoised = cv2.bilateralFilter(
        src=image,
        d=5,                    # Diameter
        sigmaColor=75,          # Color similarity sigma
        sigmaSpace=75           # Spatial sigma
    )
    RETURN denoised
```

**Alternative (for severe artifacts)**:
```
FUNCTION moderate_denoise(image: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising for stronger artifact removal.
    More computationally expensive.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        src=image,
        h=3,                    # Filter strength (luminance)
        hForColorComponents=3,  # Filter strength (color)
        templateWindowSize=7,
        searchWindowSize=21
    )
    RETURN denoised
```

#### 6.3.3 CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Purpose**: Enhance local contrast, especially useful for faces in shadow or uneven lighting.

**When to Apply**: 
- Configurable via `config.preprocessing.apply_clahe`
- Recommended for indoor gym lighting conditions

```
FUNCTION apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE to L channel of LAB color space.
    Enhances local contrast without affecting color balance.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Merge and convert back to RGB
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    RETURN enhanced_rgb
```

#### 6.3.4 Resolution Normalization

**Strategy**: Limit maximum dimension to 2048px while preserving aspect ratio.

```
FUNCTION normalize_resolution(image: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    """
    Resize image if larger than max_dim, preserving aspect ratio.
    Uses INTER_AREA for downscaling (best quality).
    """
    height, width = image.shape[:2]
    
    IF max(height, width) <= max_dim:
        RETURN image  # No resize needed
    
    # Calculate scale factor
    scale = max_dim / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=cv2.INTER_AREA  # Best for downscaling
    )
    
    RETURN resized
```

### 6.4 Full Preprocessing Pipeline

```
FUNCTION preprocess_image(
    metadata: ImageMetadata,
    config: PreprocessConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Full preprocessing pipeline for WhatsApp-compressed images.
    Returns preprocessed image and list of operations applied.
    """
    operations_applied = []
    
    # Load image
    image = Image.open(metadata.filepath)
    
    # Step 1: Color correction
    IF image.mode != 'RGB':
        image = ensure_rgb(image)
        operations_applied.append(f"color_conversion:{image.mode}->RGB")
    
    # Convert to numpy for OpenCV operations
    image_array = np.array(image)
    
    # Step 2: Resolution normalization (BEFORE other processing)
    original_shape = image_array.shape[:2]
    image_array = normalize_resolution(image_array, config.max_resolution)
    IF image_array.shape[:2] != original_shape:
        operations_applied.append(f"resize:{original_shape}->{image_array.shape[:2]}")
    
    # Step 3: Denoising
    IF config.denoise_strength > 0:
        IF config.denoise_strength <= 3:
            image_array = light_denoise(image_array)
            operations_applied.append("denoise:bilateral")
        ELSE:
            image_array = moderate_denoise(image_array)
            operations_applied.append("denoise:nlmeans")
    
    # Step 4: CLAHE (optional)
    IF config.apply_clahe:
        image_array = apply_clahe(image_array, config.clahe_clip_limit)
        operations_applied.append(f"clahe:clip={config.clahe_clip_limit}")
    
    # Update metadata
    metadata.preprocessed = True
    metadata.preprocessing_applied = operations_applied
    
    RETURN image_array, operations_applied
```

---

## 7. Ingestion Stage Output

### 7.1 Output Data Structure

```
CLASS IngestionResult:
    success: bool
    metadata: ImageMetadata
    image_array: np.ndarray           # Preprocessed image (HxWx3, RGB, uint8)
    preprocessing_log: List[str]      # Operations applied
    validation_warnings: List[str]    # Non-blocking issues
    
CLASS BatchIngestionResult:
    total_files_scanned: int
    valid_images: int
    invalid_images: int
    skipped_files: int
    results: List[IngestionResult]    # Chronologically sorted
    error_log: List[Dict]             # Detailed error information
```

### 7.2 Stage Output Flow

```
FOR each valid_image IN sorted_images:
    
    validation_result = validate_image(valid_image.metadata)
    
    IF NOT validation_result.valid:
        LOG_ERROR(f"Validation failed: {validation_result.errors}")
        CONTINUE
    
    preprocessed_image, ops = preprocess_image(
        valid_image.metadata,
        config.preprocessing
    )
    
    YIELD IngestionResult(
        success=True,
        metadata=valid_image.metadata,
        image_array=preprocessed_image,
        preprocessing_log=ops,
        validation_warnings=validation_result.warnings
    )
```

---

## 8. Error Handling Strategy

### 8.1 Error Categories

| Error Type | Example | Recovery | Logging |
|------------|---------|----------|---------|
| **File I/O** | Permission denied | Skip file, continue | ERROR |
| **Corruption** | Truncated JPEG | Skip file, continue | ERROR |
| **Validation** | Resolution too low | Process with warning | WARNING |
| **Preprocessing** | OOM during CLAHE | Retry without CLAHE | WARNING |
| **System** | Disk full | Graceful shutdown | CRITICAL |

### 8.2 Retry Logic

```
FUNCTION safe_ingest(metadata: ImageMetadata, config: Config) -> IngestionResult:
    
    max_retries = 3
    retry_count = 0
    
    WHILE retry_count < max_retries:
        TRY:
            RETURN ingest_image(metadata, config)
        EXCEPT MemoryError:
            retry_count += 1
            # Reduce preprocessing intensity
            config.preprocessing.apply_clahe = False
            config.preprocessing.denoise_strength = 0
            LOG_WARNING(f"Retry {retry_count}: Reduced preprocessing for {metadata.filename}")
        EXCEPT Exception AS e:
            LOG_ERROR(f"Ingestion failed for {metadata.filename}: {e}")
            RETURN IngestionResult(success=False, error=str(e))
    
    RETURN IngestionResult(success=False, error="Max retries exceeded")
```

---

## 9. Logging Specifications

### 9.1 Log Messages

```
# Stage start
INFO  | Starting data ingestion | input_dir=/app/images

# Scanning
INFO  | Directory scan complete | total_files=45 | valid=42 | skipped=3

# Per-file processing
DEBUG | Processing file | filename=202310271.jpg | date=2023-10-27 | session=1
INFO  | Validation passed | filename=202310271.jpg | resolution=1600x1200
DEBUG | Preprocessing complete | filename=202310271.jpg | ops=['denoise:bilateral', 'clahe:clip=2.0']

# Warnings
WARN  | Low resolution image | filename=202310273.jpg | resolution=480x360

# Errors
ERROR | Validation failed | filename=202310274.jpg | error="Corrupted JPEG header"
ERROR | File not found | filename=202310275.jpg

# Stage complete
INFO  | Data ingestion complete | processed=40 | failed=2 | duration=12.5s
```

### 9.2 Structured Logging Format

```json
{
    "timestamp": "2026-03-28T14:30:00.000Z",
    "level": "INFO",
    "stage": "ingestion",
    "event": "file_processed",
    "data": {
        "filename": "202310271.jpg",
        "date": "2023-10-27",
        "session": 1,
        "resolution": [1600, 1200],
        "preprocessing": ["denoise:bilateral"],
        "duration_ms": 245
    }
}
```

---

## 10. Configuration Reference

```yaml
# Ingestion stage configuration (subset of pipeline_config.yaml)

ingestion:
  input_dir: "/app/images"
  
  # File filtering
  file_patterns:
    - "*.jpg"
    - "*.jpeg"
  naming_convention: "^(\\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])([1-9])\\.jpe?g$"
  
  # Validation thresholds
  validation:
    min_width: 640
    min_height: 480
    max_width: 8192
    max_height: 8192
    allowed_modes: ["RGB", "RGBA", "L", "P"]
  
  # Preprocessing
  preprocessing:
    max_resolution: 2048
    denoise_strength: 3          # 0=off, 1-3=light, 4+=moderate
    apply_clahe: true
    clahe_clip_limit: 2.0
  
  # Behavior
  skip_on_error: true             # Continue processing other images on error
  max_concurrent: 1               # Sequential for memory safety
```

---

*Document Version: 1.0*  
*Last Updated: 2026-03-28*  
*Author: CV Pipeline Planning*
