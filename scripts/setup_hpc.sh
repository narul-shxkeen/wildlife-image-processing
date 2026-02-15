#!/bin/bash
# Quick setup script for HPC environment

echo "=========================================="
echo "SAM YOLOv12 Segmentation - Setup Script"
echo "=========================================="
echo ""

# Check if running on HPC (adjust this check based on your system)
if command -v module &> /dev/null; then
    echo "✅ Module system detected (HPC environment)"
    echo ""
    
    echo "Loading modules..."
    module purge
    module load python/3.10
    module load cuda/11.8
    echo "✅ Modules loaded"
    echo ""
else
    echo "⚠️  No module system detected (local environment)"
    echo ""
fi

# Create virtual environment
VENV_PATH="$HOME/venvs/sam_yolo"

if [ -d "$VENV_PATH" ]; then
    echo "⚠️  Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
    else
        echo "Using existing environment"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python -m venv "$VENV_PATH"
    echo "✅ Virtual environment created"
fi

# Activate environment
echo ""
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✅ Activated: $VIRTUAL_ENV"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r scripts/requirements.txt

# Install Segment Anything
echo ""
echo "Installing Segment Anything Model..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry

print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - will run on CPU (slow)')
"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "To use this environment:"
    echo "  source $VENV_PATH/bin/activate"
    echo ""
    echo "To run the segmentation script:"
    echo "  python scripts/sam_yolo_segmentation.py --help"
    echo ""
    echo "To submit SLURM job:"
    echo "  sbatch scripts/run_sam_segmentation.slurm"
else
    echo "=========================================="
    echo "❌ Setup failed!"
    echo "=========================================="
    echo "Please check the error messages above"
fi
