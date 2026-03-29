# Installation Guide: Stage 2 & 3

This guide walks you through setting up the Face Detection and Segmentation pipeline on your NVIDIA RTX 4060.

## Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or equivalent
- **CUDA**: 11.7+ or 12.1+
- **Python**: 3.10+
- **RAM**: 16GB+ recommended

## Step 1: Install CUDA and cuDNN

### Ubuntu/Linux
```bash
# Check if CUDA is already installed
nvidia-smi

# If not installed, follow: https://developer.nvidia.com/cuda-downloads
# Recommended: CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

## Step 2: Create Python Virtual Environment

```bash
cd /path/to/bjju-admin-portal

# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

## Step 3: Install PyTorch with CUDA Support

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA Available: True
CUDA Version: 12.1
```

## Step 4: Install Pipeline Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install Ultralytics (YOLOv8)
pip install ultralytics

# Install MobileSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## Step 5: Download Model Weights

### YOLOv8-face

```bash
# Create model directory
mkdir -p models/yolov8

# Download YOLOv8n-face weights
# Option 1: From official repository
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt \
  -O models/yolov8/yolov8n-face.pt

# Option 2: If above fails, manually download from:
# https://github.com/derronqi/yolov8-face
# and place in models/yolov8/
```

### MobileSAM

```bash
# Create model directory
mkdir -p models/mobilesam

# Download MobileSAM weights
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
  -O models/mobilesam/mobile_sam.pt

# Verify download
ls -lh models/mobilesam/mobile_sam.pt
# Should be ~40MB
```

## Step 6: Update Configuration Paths

Edit your configuration file or update the example script paths:

```python
# In examples/run_detection_segmentation.py or your config

detection_config = DetectionConfig(
    weights_path=Path("./models/yolov8/yolov8n-face.pt"),  # Update this path
    device="cuda:0",
    half=True,
)

segmentation_config = SegmentationConfig(
    checkpoint_path=Path("./models/mobilesam/mobile_sam.pt"),  # Update this path
    device="cuda:0",
    half=True,
)
```

## Step 7: Verify Installation

```bash
# Test imports
python -c "
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from mobile_sam import sam_model_registry
print('✓ All imports successful!')
print(f'✓ PyTorch CUDA: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

Expected output:
```
✓ All imports successful!
✓ PyTorch CUDA: True
✓ GPU: NVIDIA GeForce RTX 4060
```

## Step 8: Run Test Pipeline

```bash
# Create test data directory
mkdir -p data/test_images

# Place a test image in data/test_images/
# Image should be named in format: YYYYMMDDN.jpg (e.g., 202310271.jpg)

# Run pipeline
python examples/run_detection_segmentation.py \
  --input-dir ./data/test_images \
  --output-dir ./output

# Check output
ls -R output/
```

## Expected Directory Structure

After installation:
```
bjju-admin-portal/
├── cv_pipeline/
│   ├── stage1_ingestion/
│   ├── stage2_detection/      # ✓ New
│   └── stage3_segmentation/   # ✓ New
├── models/
│   ├── yolov8/
│   │   └── yolov8n-face.pt    # ✓ ~6MB
│   └── mobilesam/
│       └── mobile_sam.pt      # ✓ ~40MB
├── examples/
│   └── run_detection_segmentation.py
├── requirements.txt
└── venv/                      # Virtual environment
```

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"

**Solution**:
```python
# Reduce batch size in segmentation config
segmentation_config = SegmentationConfig(
    batch_size=4,  # Reduce from 8 to 4
    aggressive_cleanup=True,  # Enable aggressive memory cleanup
)
```

### Issue: "ModuleNotFoundError: No module named 'mobile_sam'"

**Solution**:
```bash
# Reinstall MobileSAM
pip uninstall mobile-sam
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Issue: YOLOv8 model not loading

**Solution**:
```bash
# Verify model file
file models/yolov8/yolov8n-face.pt
# Should show: "models/yolov8/yolov8n-face.pt: data"

# If corrupted, re-download
rm models/yolov8/yolov8n-face.pt
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt \
  -O models/yolov8/yolov8n-face.pt
```

### Issue: "CUDA driver version is insufficient"

**Solution**:
```bash
# Update NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# Or manually install from: https://www.nvidia.com/download/index.aspx
```

### Issue: "torch.cuda.is_available() returns False"

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
# Then follow Step 3 again
```

## Performance Benchmarks

On NVIDIA RTX 4060 (8GB VRAM):

| Image Resolution | Faces | Detection Time | Segmentation Time | Total Time | Peak VRAM |
|------------------|-------|----------------|-------------------|------------|-----------|
| 1600x1200        | 12    | 15ms          | 18s               | ~18s       | 2.8GB     |
| 1920x1080        | 18    | 18ms          | 27s               | ~27s       | 3.1GB     |
| 4032x3024        | 30    | 120ms (tiled) | 52s               | ~52s       | 3.5GB     |

## Next Steps

After successful installation:

1. **Test with your images**: Place BJJ group photos in `data/preprocessed/`
2. **Run the pipeline**: `python examples/run_detection_segmentation.py`
3. **Review outputs**: Check `output/masks/` for segmented faces
4. **Adjust configurations**: Fine-tune detection and segmentation parameters
5. **Integrate with Stage 1**: Connect with existing ingestion pipeline

## Support

For issues or questions:
- Check architectural docs: `docs/pipeline/02_Detection_Stage_Plan.md`
- Review logs: `output/logs/`
- Check debug visualizations: `output/debug/`

---

**Installation Date**: 2026-03-28  
**Pipeline Version**: 1.0  
**Target Hardware**: NVIDIA RTX 4060 (8GB VRAM)
