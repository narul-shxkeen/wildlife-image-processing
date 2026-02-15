# HPC Quick Start Guide

## Files Created

1. **`sam_yolo_segmentation.py`** - Main Python script (converted from notebook)
2. **`run_sam_segmentation.slurm`** - SLURM job submission script
3. **`setup_hpc.sh`** - Automated environment setup script
4. **`requirements.txt`** - Python dependencies
5. **`README_HPC.md`** - Comprehensive documentation

## Quick Setup (3 Steps)

### Step 1: Upload Files to HPC

```bash
# From your local machine, upload the entire scripts directory
scp -r scripts/ your_username@hpc.server.edu:~/wildLifeImageProcessing/

# Also upload your data
scp -r dataset/ your_username@hpc.server.edu:~/wildLifeImageProcessing/
scp image_categories_cleaned.json your_username@hpc.server.edu:~/wildLifeImageProcessing/
scp sam_vit_b_01ec64.pth your_username@hpc.server.edu:~/wildLifeImageProcessing/
```

### Step 2: Run Setup Script

```bash
# SSH into HPC
ssh your_username@hpc.server.edu

# Navigate to project directory
cd ~/wildLifeImageProcessing

# Run automated setup
bash scripts/setup_hpc.sh
```

### Step 3: Edit and Submit Job

```bash
# Edit the SLURM script to set your paths
nano scripts/run_sam_segmentation.slurm

# Update these lines:
#   DATASET_DIR="/path/to/dataset"           → your actual dataset path
#   LABELS_JSON="/path/to/image_categories..." → your labels file path
#   SAM_CHECKPOINT="/path/to/sam_vit_b..."   → your SAM checkpoint path
#   OUTPUT_DIR="/path/to/output_sam_yolo_${SLURM_JOB_ID}"

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/run_sam_segmentation.slurm
```

## Monitoring Your Job

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/sam_segmentation_<jobid>.out

# View errors
tail -f logs/sam_segmentation_<jobid>.err

# Cancel job if needed
scancel <jobid>
```

## Running Locally (Testing)

If you want to test the script locally first:

```bash
# Activate environment
source ~/venvs/sam_yolo/bin/activate

# Test on small sample
python scripts/sam_yolo_segmentation.py \
    --dataset ./dataset \
    --labels ./image_categories_cleaned.json \
    --output ./test_output \
    --sam-checkpoint ./sam_vit_b_01ec64.pth \
    --num-samples 10 \
    --max-image-size 640
```

## Common HPC-Specific Adjustments

### 1. Different Module Names

If your HPC uses different module names, edit `setup_hpc.sh` and `run_sam_segmentation.slurm`:

```bash
# Example alternatives:
module load python3/3.10.5
module load cuda/11.7
module load cudatoolkit/11.8
```

### 2. Different Partition Names

Edit the SLURM script `#SBATCH --partition=` line:

```bash
#SBATCH --partition=gpu      # common
#SBATCH --partition=gpu_v100  # specific GPU
#SBATCH --partition=gpuA100   # A100 GPUs
```

### 3. Memory Issues

If you get OOM errors:

```bash
# In SLURM script, increase memory:
#SBATCH --mem=64G

# And in the python command, reduce image size:
python scripts/sam_yolo_segmentation.py ... --max-image-size 512
```

## Expected Output

After successful completion:

```
output_sam_yolo_<jobid>/
├── images/          # 1200 processed images
├── labels/          # 1200 YOLO format .txt files
├── vis/             # 1200 visualization images
└── classes.txt      # List of 7 species classes
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not found | Check module load commands match your HPC |
| Job pending forever | Check partition name and resource availability |
| Out of memory | Reduce `--max-image-size` or request more memory |
| Import errors | Re-run `setup_hpc.sh` |
| Slow processing | Make sure GPU partition is requested |

## Performance Benchmarks

- **GPU (V100)**: ~3-8 seconds per image → 1200 images in 1-2 hours
- **GPU (A100)**: ~2-5 seconds per image → 1200 images in 40-90 minutes
- **CPU only**: ~30-60 seconds per image → NOT RECOMMENDED

## Support

For detailed documentation, see `README_HPC.md` in the scripts directory.
