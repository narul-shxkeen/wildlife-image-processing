# SAM YOLOv12 Segmentation Script - HPC Guide

## Overview

The `sam_yolo_segmentation.py` script converts your wildlife image dataset into a YOLOv12-compatible format using the Segment Anything Model (SAM) for automatic segmentation.

## Prerequisites

### 1. Python Environment

Create a Python environment with required packages:

```bash
# On HPC, load required modules (example for common HPC systems)
module load python/3.10
module load cuda/11.8  # or appropriate CUDA version

# Create virtual environment
python -m venv ~/venvs/sam_yolo
source ~/venvs/sam_yolo/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless pillow numpy tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Required Files

Upload these files to your HPC:
- Your `dataset/` directory with images
- `image_categories_cleaned.json` labels file
- `sam_vit_b_01ec64.pth` checkpoint (or use `--download-checkpoint` flag)

## Usage

### Basic Usage

```bash
python sam_yolo_segmentation.py \
    --dataset /path/to/dataset \
    --labels /path/to/image_categories_cleaned.json \
    --output /path/to/output_sam_yolo
```

### Full Example with All Options

```bash
python sam_yolo_segmentation.py \
    --dataset ./dataset \
    --labels ./image_categories_cleaned.json \
    --output ./output_sam_yolo \
    --sam-checkpoint ./sam_vit_b_01ec64.pth \
    --sam-model-type vit_b \
    --num-samples 1200 \
    --device cuda \
    --max-image-size 640 \
    --points-per-side 12 \
    --pred-iou-thresh 0.88 \
    --stability-score-thresh 0.95 \
    --area-threshold 0.005 \
    --seed 42
```

### Command-Line Arguments

#### Required Arguments:
- `--dataset`: Path to dataset directory containing images
- `--labels`: Path to labels JSON file (image_categories_cleaned.json)
- `--output`: Output directory for YOLO dataset

#### Optional Arguments:

**SAM Model:**
- `--sam-checkpoint`: Path to SAM checkpoint file (default: `./sam_vit_b_01ec64.pth`)
- `--sam-model-type`: SAM model type: `vit_h`, `vit_l`, or `vit_b` (default: `vit_b`)
- `--download-checkpoint`: Auto-download SAM checkpoint if not found

**Processing Parameters:**
- `--num-samples`: Number of images to sample (default: 1200)
- `--seed`: Random seed for reproducibility (default: 42)
- `--area-threshold`: Minimum mask area as fraction of image (default: 0.005)

**Device and Memory:**
- `--device`: Device to run on: `cuda` or `cpu` (default: `cuda`)
- `--max-image-size`: Maximum image long side in pixels (default: 640)

**SAM Performance Tuning:**
- `--points-per-side`: SAM points per side, lower = faster (default: 12)
- `--pred-iou-thresh`: Prediction IOU threshold (default: 0.88)
- `--stability-score-thresh`: Stability score threshold (default: 0.95)
- `--top-n-masks`: Number of largest masks to consider (default: 5)

**Species Filtering:**
- `--species`: List of species to filter (default: 7 predefined species)

### Running on HPC with SLURM

See `run_sam_segmentation.slurm` for a sample SLURM job script.

```bash
sbatch run_sam_segmentation.slurm
```

## Output Structure

The script creates the following output structure:

```
output_sam_yolo/
├── images/          # Processed images
├── labels/          # YOLO format labels (.txt files)
├── vis/             # Visualization images with bounding boxes
└── classes.txt      # List of class names
```

## Performance Optimization

### For Large Datasets:
- Use `--max-image-size 640` or lower to reduce memory usage
- Lower `--points-per-side` to 8-10 for faster processing
- Increase `--pred-iou-thresh` and `--stability-score-thresh` to filter out low-quality masks

### For Better Quality:
- Increase `--max-image-size` to 1024
- Increase `--points-per-side` to 32
- Lower thresholds for more mask candidates

### Memory Issues:
If you encounter CUDA Out of Memory errors:
1. Reduce `--max-image-size` (try 512 or 480)
2. Reduce `--points-per-side` (try 8)
3. Request more GPU memory in your SLURM script

## Expected Runtime

With GPU (NVIDIA V100 or similar):
- ~3-8 seconds per image
- 1200 images: ~1-2 hours

With CPU only:
- ~30-60 seconds per image
- 1200 images: ~10-20 hours (not recommended)

## Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### SAM Checkpoint Not Found
```bash
# Download manually
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or use auto-download flag
python sam_yolo_segmentation.py ... --download-checkpoint
```

### Import Errors
```bash
# Reinstall segment-anything
pip uninstall segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Contact

For issues or questions, refer to the main project repository.




annotation.pbs  image_categories_cleaned.json  segmentation.py  test.py
dataset         requirements.txt               setup.sh
