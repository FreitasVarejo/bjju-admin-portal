# BJJU Admin Portal - Computer Vision Pipeline
# Stage 1: Data Ingestion & Preprocessing

# Use Python 3.11 slim image for smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cv_pipeline/ /app/cv_pipeline/

# Create necessary directories for data and logs
RUN mkdir -p /app/images \
    /app/data/preprocessed \
    /app/data/logs \
    /app/data/failed \
    /app/data/intermediate

# Set Python path
ENV PYTHONPATH=/app

# Default command runs the ingestion pipeline
CMD ["python", "-m", "cv_pipeline.stage1_ingestion.ingestion"]

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD test -f /app/data/logs/*.log || exit 1

# Labels for documentation
LABEL maintainer="BJJU Admin Portal"
LABEL description="Computer Vision Pipeline - Stage 1: Data Ingestion & Preprocessing"
LABEL version="0.1.0"
