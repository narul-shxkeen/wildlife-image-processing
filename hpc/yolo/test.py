#!/usr/bin/env python3
"""
Quick test script to verify the environment is set up correctly.
Run this before submitting your HPC job to catch any issues early.
"""

import sys

print("=" * 70)
print("Environment Test for SAM YOLOv12 Segmentation")
print("=" * 70)
print()

# Test 1: Python version
print("1. Checking Python version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✅ Python version OK (>= 3.8)")
else:
    print("   ❌ Python version too old (need >= 3.8)")
    sys.exit(1)

# Test 2: Import core libraries
print("\n2. Testing core library imports...")
try:
    import numpy as np
    print(f"   ✅ numpy {np.__version__}")
except ImportError as e:
    print(f"   ❌ numpy import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"   ✅ opencv {cv2.__version__}")
except ImportError as e:
    print(f"   ❌ opencv import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"   ✅ Pillow {Image.__version__}")
except ImportError as e:
    print(f"   ❌ Pillow import failed: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    print(f"   ✅ tqdm")
except ImportError as e:
    print(f"   ❌ tqdm import failed: {e}")
    sys.exit(1)

# Test 3: PyTorch and CUDA
print("\n3. Testing PyTorch and CUDA...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.version.cuda}")
        print(f"   ✅ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"   ✅ Successfully created tensor on GPU: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Failed to create tensor on GPU: {e}")
            sys.exit(1)
    else:
        print(f"   ⚠️  CUDA NOT available - will run on CPU (very slow)")
        print("   Possible fixes:")
        print("   - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("   - Load CUDA module on HPC: module load cuda/11.8")
        
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")
    sys.exit(1)

# Test 4: Segment Anything
print("\n4. Testing Segment Anything Model...")
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print(f"   ✅ segment_anything imported successfully")
    print(f"   ✅ Available SAM models: {list(sam_model_registry.keys())}")
except ImportError as e:
    print(f"   ❌ segment_anything import failed: {e}")
    print("   Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

# Test 5: Check for required files (optional)
print("\n5. Checking for required files (optional)...")
from pathlib import Path

files_to_check = [
    ('scripts/sam_yolo_segmentation.py', 'Main script'),
    ('scripts/run_sam_segmentation.slurm', 'SLURM job script'),
    ('dataset/', 'Dataset directory'),
    ('image_categories_cleaned.json', 'Labels file'),
    ('sam_vit_b_01ec64.pth', 'SAM checkpoint'),
]

for file_path, description in files_to_check:
    p = Path(file_path)
    if p.exists():
        if p.is_dir():
            count = len(list(p.glob('*.jpg'))) + len(list(p.glob('*.JPG'))) + len(list(p.glob('*.png')))
            print(f"   ✅ {description}: {file_path} ({count} images)")
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"   ✅ {description}: {file_path} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠️  {description} not found: {file_path}")

# Test 6: Test script syntax
print("\n6. Testing main script syntax...")
try:
    with open('scripts/sam_yolo_segmentation.py', 'r') as f:
        compile(f.read(), 'scripts/sam_yolo_segmentation.py', 'exec')
    print("   ✅ Main script syntax is valid")
except Exception as e:
    print(f"   ❌ Syntax error in main script: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print("✅ All critical tests passed!")
print("\nYour environment is ready to run the segmentation pipeline.")
print("\nNext steps:")
print("  1. Edit scripts/run_sam_segmentation.slurm with your paths")
print("  2. Submit job: sbatch scripts/run_sam_segmentation.slurm")
print("  3. Monitor: tail -f logs/sam_segmentation_<jobid>.out")
print("=" * 70)

