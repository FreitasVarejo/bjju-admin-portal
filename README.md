# BJJU Admin Portal - Computer Vision Pipeline

Automated computer vision pipeline for processing group photos of Jiu-Jitsu practitioners. The system extracts and segments individual faces (including hair) to produce clean, isolated masks suitable for downstream facial recognition and attendance tracking.

## 🎯 Project Overview

This project implements a three-stage pipeline:

1. **Stage 1**: Data Ingestion & Preprocessing
   - Scans and validates image files
   - Parses metadata from filenames (YYYYMMDDH format)
   - Preprocesses images for model inference

2. **Stage 2**: Face Detection (YOLOv8-face)
   - Detects all visible faces in group photos
   - Extracts bounding boxes with confidence scores
   - Optimized for high-resolution smartphone images

3. **Stage 3**: Zero-Shot Segmentation (MobileSAM)
   - Generates precise face+hair segmentation masks
   - Uses bounding box prompts for accurate results
   - Produces clean PNG masks with black backgrounds

## 🏗️ System Architecture

```
INPUT (./images/) 
    ↓
STAGE 1: Data Ingestion & Preprocessing
    ↓
STAGE 2: YOLOv8-face Detection
    ↓
STAGE 3: MobileSAM Segmentation
    ↓
OUTPUT (./output/masks/)
```

## 📋 Requirements

- **GPU**: NVIDIA GPU with CUDA 12.9 support (tested on RTX 4060 8GB)
- **Docker**: NVIDIA Container Toolkit for containerized execution
- **Memory**: 8GB VRAM minimum
- **Software**:
  - Python 3.10+
  - PyTorch with CUDA support
  - YOLOv8
  - Mobile SAM (MobileSAM)
  - OpenCV

## 🚀 Quick Start

### Stage 1: Data Ingestion & Preprocessing

```bash
# Install dependencies
pip install -r requirements.txt

# Place images in data/raw/ directory
# Images must follow naming pattern: YYYYMMDD{session}.jpg
# Example: 20260315_1.jpg

# Run Stage 1 pipeline
./run_stage1.sh

# Or run directly
python -m cv_pipeline.stage1_ingestion.ingestion
```

### Using Docker (Recommended)

```bash
# Build container
docker build -t bjju-cv-pipeline:stage1 .

# Run Stage 1
docker run -v $(pwd)/data/raw:/app/images \
           -v $(pwd)/data:/app/data \
           bjju-cv-pipeline:stage1
```

### Local Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Stage 1 pipeline
python -m cv_pipeline.stage1_ingestion.ingestion
```

## 📁 Project Structure

```
bjju-admin-portal/
├── cv_pipeline/
│   ├── stage1_ingestion/        # Stage 1: Data ingestion & preprocessing
│   │   ├── models.py            # Data models and structures
│   │   ├── scanner.py           # File discovery and parsing
│   │   ├── validator.py         # Image validation
│   │   ├── preprocessor.py      # WhatsApp-focused preprocessing
│   │   ├── ingestion.py         # Main orchestrator
│   │   └── logger.py            # Structured logging
│   ├── config/
│   │   └── pipeline_config.yaml # Central configuration
│   ├── utils/
│   │   └── exceptions.py        # Custom exceptions
│   └── README.md                # Stage 1 documentation
├── data/
│   ├── raw/                     # Input images (WhatsApp photos)
│   ├── preprocessed/            # Stage 1 output
│   ├── logs/                    # Processing logs
│   └── failed/                  # Failed images for debugging
├── tests/
│   └── test_stage1/             # Unit tests for Stage 1
├── docs/                        # Project documentation
├── media/                       # Reference images and screenshots
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── run_stage1.sh                # Stage 1 execution script
├── .gitignore                   # Git ignore configuration
└── README.md                    # This file
```

## 📚 Documentation

- **[Project Overview](docs/Project_Overview.md)** - Comprehensive architecture, memory management strategy, and design decisions
- **[Environment Setup](docs/Environment_Setup.md)** - Detailed setup instructions for local and Docker environments
- **[Roadmap & Next Steps](docs/Roadmap_Next_Steps.md)** - Future enhancements and development roadmap
- **[Stage 1 Documentation](cv_pipeline/README.md)** - Complete guide for Stage 1 ingestion and preprocessing

## 🔑 Key Features

- ✅ **Zero-Shot Segmentation**: No custom training required, uses foundation models (SAM)
- ✅ **Memory Efficient**: Sequential model loading optimized for 8GB VRAM
- ✅ **Graceful Degradation**: Continues processing even if individual images fail
- ✅ **Comprehensive Logging**: Complete audit trail of processing decisions
- ✅ **Idempotent Processing**: Safe to rerun on the same inputs
- ✅ **High Accuracy**: Combines YOLOv8's precise detections with SAM's segmentation

## ⚙️ Configuration

### Input Specifications

- **Format**: JPEG images
- **Naming**: `YYYYMMDDH.jpg` (e.g., `202303281.jpg`)
- **Location**: `./images/` directory (recursive scanning)
- **Resolution**: Recommended 2048px max on longest edge

### Output Specifications

- **Format**: PNG (lossless)
- **Background**: Solid black (RGB: 0,0,0)
- **Structure**: Organized by date and session
- **Naming**: `{stem}_face_{idx:03d}_{bbox_hash}.png`

## 📊 Performance Benchmarks

| Stage | Time | VRAM |
|-------|------|------|
| Data Loading | 2-5s | 0.5GB |
| YOLOv8 Detection | 5-15s | 1.5GB |
| Model Swap | 3-5s | - |
| SAM Segmentation | ~50ms/face | 1.5GB |
| Post-processing | 5-10s | 0.5GB |

**Total**: ~70s per image (comfortably within 2-minute budget)

## 🛠️ Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Face Detection | YOLOv8-face | Accurate, fast, widely supported |
| Segmentation | MobileSAM | Lower VRAM, good accuracy-speed tradeoff |
| Container | Docker + NVIDIA | Consistent GPU environment |
| Framework | PyTorch | Industry standard for computer vision |

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing conventions
- Changes are well-documented
- Testing is included where applicable

## 📝 License

[Specify your license here]

## 👤 Author

Developed as an automated attendance tracking solution for Jiu-Jitsu practitioners.

---

**Last Updated**: March 28, 2026  
**Version**: 1.0
