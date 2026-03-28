# Environment Setup Plan

This document details the containerized development environment setup for the Jiu-Jitsu Attendance CV Pipeline, ensuring reproducible deployments with GPU acceleration.

---

## 1. Docker Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST SYSTEM                                  │
│                        Fedora 41                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              NVIDIA Container Toolkit                          │ │
│  │                  (nvidia-docker2)                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              DOCKER CONTAINER                                   │ │
│  │         bjju-cv-pipeline:latest                                │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │                                                                 │ │
│  │  Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04           │ │
│  │        (Closest compatible image to CUDA 12.9)                 │ │
│  │                                                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│  │  │  Python     │  │  PyTorch    │  │  CV Models  │            │ │
│  │  │  3.11       │  │  2.2+cu121  │  │  YOLO/SAM   │            │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│  │                                                                 │ │
│  │  VOLUME MOUNTS:                                                │ │
│  │  ├── ./images/  → /app/images/   (input, read-only)           │ │
│  │  ├── ./output/  → /app/output/   (output, read-write)         │ │
│  │  └── ./models/  → /app/models/   (model cache)                │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              NVIDIA RTX 4060 (8GB VRAM)                        │ │
│  │                    GPU Passthrough                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Host Prerequisites

### 2.1 NVIDIA Driver Installation (Fedora 41)

**Verification Commands:**
```bash
# Check current driver version
nvidia-smi

# Expected output should show:
# - Driver Version: 550.x or higher
# - CUDA Version: 12.x
```

**If driver installation needed:**
```bash
# Enable RPM Fusion repositories
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install NVIDIA driver
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda

# Reboot required
sudo systemctl reboot
```

### 2.2 NVIDIA Container Toolkit Installation

```bash
# Configure the repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
sudo dnf install nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 3. Docker Implementation Plan

### 3.1 Base Image Selection

| Image Option | CUDA | Size | Compatibility | Selection |
|-------------|------|------|---------------|-----------|
| `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | 12.4 | ~3.5GB | Excellent | **RECOMMENDED** |
| `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | 12.1 | ~3.2GB | Good | Alternative |
| `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` | 12.1 | ~6GB | Good | If faster setup needed |

**Rationale for CUDA 12.4.1**: 
- Host CUDA 12.9 is backward compatible with container CUDA 12.4
- PyTorch 2.2+ with `cu121` wheels work seamlessly
- Smaller image than pre-built PyTorch images
- Better control over Python environment

### 3.2 Dockerfile Plan

```dockerfile
# =============================================================================
# BJJU CV Pipeline - Dockerfile
# =============================================================================
# Build: docker build -t bjju-cv-pipeline:latest .
# Run:   docker run --gpus all -v ./images:/app/images:ro \
#                              -v ./output:/app/output \
#                              bjju-cv-pipeline:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base CUDA Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# -----------------------------------------------------------------------------
# Stage 2: Python Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Application
# -----------------------------------------------------------------------------
FROM dependencies AS app

# Create non-root user for security
RUN useradd -m -u 1000 cvuser
USER cvuser

WORKDIR /app

# Copy application code
COPY --chown=cvuser:cvuser src/ ./src/
COPY --chown=cvuser:cvuser config/ ./config/

# Create volume mount points
RUN mkdir -p /app/images /app/output /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models/huggingface

# Default command
CMD ["python", "src/main.py"]
```

### 3.3 Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  cv-pipeline:
    build:
      context: .
      dockerfile: Dockerfile
    image: bjju-cv-pipeline:latest
    container_name: bjju-cv-pipeline
    
    # GPU Configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Volume Mounts
    volumes:
      - ./images:/app/images:ro          # Input images (read-only)
      - ./output:/app/output:rw          # Output masks (read-write)
      - ./models:/app/models:rw          # Model cache (persistent)
      - ./logs:/app/logs:rw              # Application logs
    
    # Environment Variables
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - PYTHONUNBUFFERED=1
    
    # Resource Limits
    shm_size: '2gb'                       # Shared memory for PyTorch DataLoader
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
```

---

## 4. Volume Mapping Strategy

### 4.1 Volume Specifications

| Host Path | Container Path | Mode | Purpose |
|-----------|---------------|------|---------|
| `./images/` | `/app/images/` | `ro` (read-only) | Input JPEG images |
| `./output/` | `/app/output/` | `rw` (read-write) | Segmented face masks |
| `./models/` | `/app/models/` | `rw` (read-write) | Cached model weights |
| `./logs/` | `/app/logs/` | `rw` (read-write) | Processing logs |
| `./config/` | `/app/config/` | `ro` (read-only) | Configuration files |

### 4.2 Directory Structure on Host

```
bjju-admin-portal/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── images/                    # INPUT: Group photos
│   ├── 202310271.jpg
│   ├── 202310272.jpg
│   └── ...
├── output/                    # OUTPUT: Processed masks
│   ├── masks/
│   ├── metadata/
│   └── debug/
├── models/                    # CACHE: Downloaded model weights
│   ├── yolov8/
│   ├── mobilesam/
│   └── huggingface/
├── logs/                      # LOGS: Processing logs
│   └── pipeline.log
├── config/                    # CONFIG: Pipeline configuration
│   └── pipeline_config.yaml
├── src/                       # SOURCE: Application code
│   ├── main.py
│   ├── ingestion/
│   ├── detection/
│   └── segmentation/
└── docs/                      # DOCUMENTATION
    └── ...
```

---

## 5. Python Dependencies

### 5.1 requirements.txt

```txt
# =============================================================================
# BJJU CV Pipeline - Python Dependencies
# =============================================================================
# Install with: pip install -r requirements.txt
# =============================================================================

# -----------------------------------------------------------------------------
# Core ML Framework
# -----------------------------------------------------------------------------
torch==2.2.1+cu121
torchvision==0.17.1+cu121
--extra-index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------------
# Computer Vision
# -----------------------------------------------------------------------------
opencv-python-headless==4.9.0.80      # Headless for Docker (no GUI)
Pillow==10.2.0                         # Image I/O
scikit-image==0.22.0                   # Image processing utilities

# -----------------------------------------------------------------------------
# Detection: YOLOv8
# -----------------------------------------------------------------------------
ultralytics==8.1.0                     # YOLOv8 framework

# -----------------------------------------------------------------------------
# Segmentation: SAM
# -----------------------------------------------------------------------------
segment-anything==1.0                  # Original SAM (for reference)
mobile-sam==1.0                        # MobileSAM (preferred)
# Note: Install from git if mobile-sam not on PyPI
# git+https://github.com/ChaoningZhang/MobileSAM.git

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
numpy==1.26.4                          # Array operations
scipy==1.12.0                          # Scientific computing
tqdm==4.66.2                           # Progress bars
pyyaml==6.0.1                          # Configuration parsing
python-dotenv==1.0.1                   # Environment variable management

# -----------------------------------------------------------------------------
# Logging & Monitoring
# -----------------------------------------------------------------------------
loguru==0.7.2                          # Advanced logging
rich==13.7.0                           # Rich console output

# -----------------------------------------------------------------------------
# Image Quality & Preprocessing
# -----------------------------------------------------------------------------
# For WhatsApp artifact handling
opencv-contrib-python-headless==4.9.0.80   # Additional CV modules (CLAHE, etc.)

# -----------------------------------------------------------------------------
# Development & Testing (Optional)
# -----------------------------------------------------------------------------
# pytest==8.0.2
# pytest-cov==4.1.0
# black==24.2.0
# isort==5.13.2
# mypy==1.8.0
```

### 5.2 Dependency Installation Notes

**MobileSAM Installation** (if not available via pip):
```bash
# Clone and install MobileSAM
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM
pip install -e .
```

**YOLOv8-face Weights**:
```bash
# Download pre-trained face detection weights
# Option 1: Official Ultralytics with face-specific model
# Option 2: Community weights from https://github.com/derronqi/yolov8-face
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt -P models/yolov8/
```

---

## 6. GPU Passthrough Configuration

### 6.1 NVIDIA Container Runtime Verification

```bash
# Verify NVIDIA runtime is available
docker info | grep -i nvidia

# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 6.2 GPU Memory Allocation Settings

**Environment Variables for Container**:
```bash
# Limit PyTorch memory allocation fragmentation
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Force specific GPU (if multiple)
CUDA_VISIBLE_DEVICES=0

# Enable TF32 for tensor cores (RTX 4060 supports)
NVIDIA_TF32_OVERRIDE=1
```

### 6.3 Runtime GPU Selection

```bash
# Run with specific GPU
docker run --gpus '"device=0"' bjju-cv-pipeline:latest

# Run with all GPUs (default)
docker run --gpus all bjju-cv-pipeline:latest

# Run with GPU memory limit (experimental)
docker run --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  bjju-cv-pipeline:latest
```

---

## 7. Configuration Management

### 7.1 Pipeline Configuration File

```yaml
# config/pipeline_config.yaml
# =============================================================================
# BJJU CV Pipeline Configuration
# =============================================================================

pipeline:
  name: "bjju-attendance-cv"
  version: "0.1.0"
  
paths:
  input_dir: "/app/images"
  output_dir: "/app/output"
  model_cache: "/app/models"
  log_dir: "/app/logs"

detection:
  model: "yolov8n-face"
  weights_path: "/app/models/yolov8/yolov8n-face.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 100
  input_size: 640
  use_fp16: true

segmentation:
  model: "mobilesam"
  weights_path: "/app/models/mobilesam/mobile_sam.pt"
  use_fp16: true
  batch_size: 8
  include_hair: true
  background_color: [0, 0, 0]  # RGB black

preprocessing:
  max_resolution: 2048
  denoise_strength: 3
  apply_clahe: true
  clahe_clip_limit: 2.0

output:
  format: "png"
  min_face_dimension: 112
  save_debug_visualizations: false

performance:
  max_processing_time_seconds: 120
  enable_memory_optimization: true
  force_sequential_models: true

logging:
  level: "INFO"
  format: "{time} | {level} | {message}"
  rotation: "100 MB"
```

---

## 8. Build and Run Commands

### 8.1 Quick Start

```bash
# Navigate to project directory
cd bjju-admin-portal

# Build the Docker image
docker build -t bjju-cv-pipeline:latest .

# Create required directories
mkdir -p images output models logs

# Run the pipeline
docker run --gpus all \
  -v $(pwd)/images:/app/images:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  bjju-cv-pipeline:latest
```

### 8.2 Using Docker Compose

```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 8.3 Development Mode

```bash
# Mount source code for live development
docker run --gpus all \
  -v $(pwd)/images:/app/images:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/src:/app/src:ro \
  -it bjju-cv-pipeline:latest \
  /bin/bash
```

---

## 9. Troubleshooting Guide

### 9.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| GPU not visible | `nvidia-smi` fails in container | Verify NVIDIA Container Toolkit installation |
| OOM errors | `CUDA out of memory` | Enable `force_sequential_models`, reduce batch size |
| Slow first run | Long startup time | Model weights downloading; subsequent runs use cache |
| Permission denied | Cannot write to output | Check volume mount permissions, run with `--user $(id -u):$(id -g)` |

### 9.2 Verification Commands

```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check container GPU access
docker run --rm --gpus all bjju-cv-pipeline:latest python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version in container
docker run --rm --gpus all bjju-cv-pipeline:latest python -c "import torch; print(torch.version.cuda)"
```

---

*Document Version: 1.0*  
*Last Updated: 2026-03-28*  
*Author: CV Pipeline Planning*
