#!/bin/bash
#SBATCH --job-name=sam3_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# One-time setup for SAM 3 environment on Unity HPC

set -e

echo "========================================"
echo "SAM 3 Environment Setup"
echo "Started: $(date)"
echo "========================================"

# 1. Load modules
module purge
module load cuda/12.6
module load conda/latest

# 2. Create conda environment
echo ""
echo "Step 1: Creating conda environment..."

if conda env list | grep -q "^sam3 "; then
    echo "⚠️  Environment 'sam3' already exists"
    echo "To recreate: conda env remove -n sam3 -y"
    conda activate sam3
else
    conda create -n sam3 python=3.12 -y
    conda activate sam3
    echo "✓ Environment created"
fi

# 3. Install PyTorch with CUDA 12.6
echo ""
echo "Step 2: Installing PyTorch..."
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# 4. Clone and install SAM 3
echo ""
echo "Step 3: Installing SAM 3..."

SAM3_DIR="/home/zihara_delgado_uri_edu/sam3"

if [ ! -d "$SAM3_DIR" ]; then
    cd /home/zihara_delgado_uri_edu/
    git clone https://github.com/facebookresearch/sam3.git
fi

cd "$SAM3_DIR"
pip install -e .

# Fix known dependency issues
pip install "setuptools<70"
pip install einops psutil

# 5. Install pipeline requirements
echo ""
echo "Step 4: Installing pipeline dependencies..."

PIPELINE_DIR="/home/zihara_delgado_uri_edu/sam3-pipeline"
cd "$PIPELINE_DIR"

# Install from requirements.txt
pip install -r requirements.txt

# 6. Verify installation
echo ""
echo "Step 5: Verifying installation..."

python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

try:
    from sam3 import build_sam3_image_model
    print("✓ SAM 3 import: OK")
except ImportError as e:
    print(f"❌ SAM 3 import failed: {e}")
    exit(1)

try:
    from PIL import Image
    import numpy as np
    import cv2
    print("✓ Image processing libraries: OK")
except ImportError as e:
    print(f"❌ Image libraries failed: {e}")
    exit(1)

print("\n✅ All checks passed!")
EOF

echo ""
echo "========================================"
echo "✅ Setup Complete"
echo "========================================"
echo "Conda environment: sam3"
echo "Finished: $(date)"
echo ""
echo "Next steps:"
echo "1. Download SAM 3 checkpoint: bash scripts/download_sam3.sh"
echo "2. Put data in Google Drive: queue/"
echo "3. Run pipeline: bash scripts/sync_and_run.sh"
echo "========================================"
