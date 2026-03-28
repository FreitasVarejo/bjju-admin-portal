#!/bin/bash
# Sample script to run Stage 1 ingestion pipeline

set -e  # Exit on error

echo "=========================================="
echo "BJJU CV Pipeline - Stage 1 Ingestion"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set paths (customize these)
CONFIG_PATH="cv_pipeline/config/pipeline_config.yaml"
INPUT_DIR="data/raw"
OUTPUT_DIR="data/preprocessed"

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# Create input directory if it doesn't exist
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if there are images to process
IMAGE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "Warning: No images found in $INPUT_DIR"
    echo "Please add images matching pattern: YYYYMMDD{session}.jpg"
    echo "Example: 20260315_1.jpg"
    exit 0
fi

echo "Found $IMAGE_COUNT image(s) to process"
echo ""

# Run the pipeline
echo "Starting ingestion pipeline..."
python -m cv_pipeline.stage1_ingestion.ingestion "$CONFIG_PATH"

echo ""
echo "=========================================="
echo "Pipeline execution complete!"
echo "Check logs in: data/logs/"
echo "Preprocessed images in: $OUTPUT_DIR"
echo "=========================================="
