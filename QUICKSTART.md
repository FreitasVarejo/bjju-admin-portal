# Quick Start Guide - Stage 1 Ingestion

## Prerequisites

- Python 3.11+ installed
- pip package manager

## Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Prepare Your Images (2 minutes)

1. Create the input directory:
   ```bash
   mkdir -p data/raw
   ```

2. Copy your WhatsApp group photos to `data/raw/`

3. **IMPORTANT**: Rename images to match the required format:
   - Pattern: `YYYYMMDD{session}.jpg`
   - Examples:
     - `20260315_1.jpg` → March 15, 2026, Session 1
     - `20260315_2.jpg` → March 15, 2026, Session 2
     - `20260420_1.jpg` → April 20, 2026, Session 1

## Run the Pipeline (30 seconds)

### Option 1: Using the run script (easiest)
```bash
./run_stage1.sh
```

### Option 2: Direct Python execution
```bash
python -m cv_pipeline.stage1_ingestion.ingestion
```

### Option 3: With custom config
```bash
python -m cv_pipeline.stage1_ingestion.ingestion /path/to/custom_config.yaml
```

## Check Results (1 minute)

After processing completes, check:

1. **Preprocessed images**: `data/preprocessed/`
   - Contains processed JPEG files
   - Same filenames as originals

2. **Metadata**: `data/preprocessed/batch_metadata.json`
   - Processing statistics
   - Success/failure rates
   - Individual image metadata

3. **Logs**: `data/logs/`
   - Detailed processing logs
   - Error messages for failed images

4. **Failed images**: `data/failed/`
   - Images that couldn't be processed
   - Review for debugging

## Example Output

```
==========================================
BJJU CV Pipeline - Stage 1 Ingestion
==========================================

Found 15 image(s) to process

Starting ingestion pipeline...
Processing [1/15]: 20260315_1.jpg
Successfully processed 20260315_1.jpg in 2.34s
Processing [2/15]: 20260315_2.jpg
Successfully processed 20260315_2.jpg in 2.18s
...

==========================================
Pipeline Processing Complete
==========================================
Total Images Found: 15
Successfully Processed: 14
Failed: 1
Skipped: 0
Success Rate: 93.33%
Total Processing Time: 34.56s
Average Time per Image: 2.47s
==========================================
```

## Common Issues & Solutions

### Issue: "No images found to process"
**Solution**: Ensure images are in `data/raw/` and follow naming pattern `YYYYMMDD{session}.jpg`

### Issue: "Validation failed: dimensions too small"
**Solution**: Images must be at least 640x480 pixels. Check `min_width` and `min_height` in config.

### Issue: "Processing time exceeds threshold"
**Solution**: 
- Reduce `max_dimension` in config (default: 2048)
- Disable CLAHE if not needed
- Check system resources

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: Reinstall OpenCV: `pip install opencv-python==4.9.0.80`

## Next Steps

After Stage 1 completes successfully:

1. Review preprocessed images in `data/preprocessed/`
2. Check `batch_metadata.json` for processing statistics
3. Proceed to Stage 2: Face Detection (coming soon)

## Configuration Customization

Edit `cv_pipeline/config/pipeline_config.yaml` to adjust:

- **Input/Output paths**
- **Validation constraints** (dimensions, aspect ratio)
- **Preprocessing parameters** (bilateral filter, CLAHE)
- **Logging level and format**
- **Performance settings**

See the main README in `cv_pipeline/README.md` for detailed configuration options.

## Getting Help

1. Check logs in `data/logs/` for detailed error messages
2. Review `cv_pipeline/README.md` for comprehensive documentation
3. Run tests to verify installation: `pytest tests/test_stage1/`

## Docker Alternative

If you prefer Docker:

```bash
# Build image
docker build -t bjju-cv-pipeline:stage1 .

# Run container
docker run -v $(pwd)/data/raw:/app/images \
           -v $(pwd)/data:/app/data \
           bjju-cv-pipeline:stage1
```

---

**Estimated Total Time**: ~10 minutes setup + processing time (2-5s per image)

**Support**: See main README.md for contact information
