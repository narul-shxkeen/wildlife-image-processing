#!/usr/bin/env python3
"""
SAM to YOLOv12 Dataset Generator
================================

This script samples images from a dataset, runs the Segment Anything Model (SAM) 
to generate masks, extracts bounding boxes, and creates a YOLOv12-style dataset.

Usage:
    python sam_yolo_segmentation.py --dataset dataset/ --labels image_categories_cleaned.json --output output_sam_yolo

Requirements:
    - PyTorch with CUDA support
    - segment-anything
    - opencv-python
    - tqdm
"""

import json
import random
import argparse
import urllib.request
from pathlib import Path
import shutil
import sys
import os

import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm

# Import SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError as e:
    print('Error: segment_anything not installed. Install with:')
    print('pip install git+https://github.com/facebookresearch/segment-anything.git')
    sys.exit(1)


# ============================================================================
# Helper Functions
# ============================================================================

def xyxy_to_yolo(box, img_w, img_h):
    """Convert [x_min, y_min, x_max, y_max] to YOLO format [x_center, y_center, width, height] (normalized)."""
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    return [x_center / img_w, y_center / img_h, width / img_w, height / img_h]


def bbox_center(bbox):
    """Return (x_center, y_center) of a bbox [x_min, y_min, x_max, y_max]."""
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def distance_to_image_center(bbox, img_w, img_h):
    """Compute Euclidean distance from bbox center to image center."""
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0
    bbox_cx, bbox_cy = bbox_center(bbox)
    return ((bbox_cx - img_center_x) ** 2 + (bbox_cy - img_center_y) ** 2) ** 0.5


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def check_cuda_availability():
    """Check and print CUDA availability information."""
    print('=' * 70)
    print('GPU/CUDA Status')
    print('=' * 70)
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        print(f'Current device: {torch.cuda.current_device()}')
        print('✅ CUDA is available - will use GPU acceleration')
    else:
        print('⚠️  CUDA NOT AVAILABLE - will run on CPU (very slow)')
        print('Possible reasons:')
        print('  1. PyTorch CPU-only version installed')
        print('  2. CUDA drivers not installed')
        print('  3. GPU not detected')
        print('\nTo fix, install PyTorch with CUDA:')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
    print('=' * 70)
    print()


# ============================================================================
# Main Processing Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate YOLOv12 dataset using SAM segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels JSON file (image_categories_cleaned.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for YOLO dataset')
    
    # SAM model arguments
    parser.add_argument('--sam-checkpoint', type=str, default='./sam_vit_b_01ec64.pth',
                        help='Path to SAM checkpoint file')
    parser.add_argument('--sam-model-type', type=str, default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--download-checkpoint', action='store_true',
                        help='Auto-download SAM checkpoint if not found')
    
    # Processing parameters
    parser.add_argument('--num-samples', type=int, default=6000,
                        help='Number of images to sample for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--area-threshold', type=float, default=0.005,
                        help='Minimum mask area as fraction of image area')
    
    # Device and memory arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run SAM on (cuda or cpu)')
    parser.add_argument('--max-image-size', type=int, default=640,
                        help='Maximum long side of image (reduces GPU memory usage)')
    
    # SAM performance tuning
    parser.add_argument('--points-per-side', type=int, default=12,
                        help='SAM points per side (lower = faster, default=32)')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88,
                        help='SAM prediction IOU threshold (higher = fewer masks)')
    parser.add_argument('--stability-score-thresh', type=float, default=0.95,
                        help='SAM stability score threshold (higher = fewer masks)')
    parser.add_argument('--top-n-masks', type=int, default=5,
                        help='Number of largest masks to consider for center selection')
    
    # Species filtering
    parser.add_argument('--species', type=str, nargs='+', default=None,
                        help='List of species to filter (if not provided, uses default 7 species)')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    DATASET_DIR = Path(args.dataset)
    LABELS_JSON = Path(args.labels)
    SAM_CHECKPOINT = Path(args.sam_checkpoint)
    OUTPUT_DIR = Path(args.output)
    
    # Default species list
    if args.species is None:
        ALLOWED_SPECIES = {
            "Common leopard",
            "Himalayan goral",
            "Rhesus macaque",
            "Himalayan gray langur",
            "Himalayan tahr",
            "Yellow-throated marten",
            "Leopard cat"
        }
    else:
        ALLOWED_SPECIES = set(args.species)
    
    # SAM download URL
    SAM_VIT_B_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    
    # Check CUDA availability
    check_cuda_availability()
    
    # Verify CUDA if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            '❌ CUDA not available! You requested GPU but PyTorch cannot find CUDA.\n\n'
            'Solutions:\n'
            '1. Install PyTorch with CUDA support:\n'
            '   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n\n'
            '2. Or use --device cpu (will be VERY slow)\n\n'
            f'Current PyTorch: {torch.__version__}\n'
            f'CUDA available: {torch.cuda.is_available()}'
        )
    
    # Print configuration
    print('Configuration:')
    print(f'  Dataset directory: {DATASET_DIR}')
    print(f'  Labels JSON: {LABELS_JSON}')
    print(f'  Output directory: {OUTPUT_DIR}')
    print(f'  SAM checkpoint: {SAM_CHECKPOINT}')
    print(f'  Number of samples: {args.num_samples}')
    print(f'  Device: {args.device}')
    print(f'  Max image size: {args.max_image_size}')
    print(f'  SAM performance: points_per_side={args.points_per_side}, '
          f'pred_iou={args.pred_iou_thresh}, stability={args.stability_score_thresh}')
    print(f'  Filtering to {len(ALLOWED_SPECIES)} species: {", ".join(sorted(ALLOWED_SPECIES))}')
    print()
    
    # ========================================================================
    # Validate inputs and download checkpoint if needed
    # ========================================================================
    
    if args.download_checkpoint and not SAM_CHECKPOINT.exists():
        print(f'SAM checkpoint {SAM_CHECKPOINT} not found locally.')
        print(f'Downloading from {SAM_VIT_B_URL}...')
        try:
            SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SAM_VIT_B_URL, str(SAM_CHECKPOINT))
            print('✅ Downloaded SAM checkpoint')
        except Exception as e:
            raise RuntimeError(f'Failed to download SAM checkpoint: {e}')
    
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f'Dataset directory not found: {DATASET_DIR}')
    if not LABELS_JSON.exists():
        raise FileNotFoundError(f'Labels JSON not found: {LABELS_JSON}')
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f'SAM checkpoint not found: {SAM_CHECKPOINT}')
    
    # Create output directories
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR / 'images')
    ensure_dir(OUTPUT_DIR / 'labels')
    ensure_dir(OUTPUT_DIR / 'vis')
    
    # ========================================================================
    # Load labels and filter images
    # ========================================================================
    
    print('Loading labels...')
    with open(LABELS_JSON, 'r', encoding='utf-8') as f:
        image_categories = json.load(f)
    
    # Gather images
    print('Gathering images from dataset...')
    all_images = [
        p.name for p in DATASET_DIR.iterdir() 
        if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    available_images = [img for img in all_images if img in image_categories]
    
    if not available_images:
        raise RuntimeError('No labeled images found in the dataset folder')
    
    print(f'Found {len(available_images)} labeled images')
    
    # Filter to only allowed species
    filtered_images = []
    species_counts = {sp: 0 for sp in ALLOWED_SPECIES}
    
    for img in available_images:
        cats = image_categories.get(img)
        if not cats:
            continue
        label = cats[0] if isinstance(cats, list) else cats
        if label in ALLOWED_SPECIES:
            filtered_images.append(img)
            species_counts[label] += 1
    
    print(f'Found {len(filtered_images)} images with allowed species')
    print('Species distribution:')
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
        print(f'  {species}: {count} images')
    
    # Sample images
    NUM_SAMPLES_ACTUAL = min(args.num_samples, len(filtered_images))
    random.seed(args.seed)
    sampled = random.sample(filtered_images, NUM_SAMPLES_ACTUAL)
    print(f'\n✅ Sampled {NUM_SAMPLES_ACTUAL} images for processing')
    
    # Create species to class ID mapping
    species_set = set()
    for img in sampled:
        cats = image_categories.get(img)
        if not cats:
            continue
        species_set.add(cats[0] if isinstance(cats, list) else cats)
    
    species_list = sorted(list(species_set))
    species_to_id = {s: i for i, s in enumerate(species_list)}
    
    # Write classes.txt
    with open(OUTPUT_DIR / 'classes.txt', 'w', encoding='utf-8') as f:
        for s in species_list:
            f.write(s + '\n')
    print(f'✅ Created classes.txt with {len(species_list)} classes')
    
    # ========================================================================
    # Load SAM model
    # ========================================================================
    
    print('\nLoading SAM model...')
    try:
        sam = sam_model_registry[args.sam_model_type](checkpoint=str(SAM_CHECKPOINT))
    except KeyError:
        raise KeyError(
            f'SAM model type {args.sam_model_type} not found; '
            f'available: {list(sam_model_registry.keys())}'
        )
    except Exception as e:
        raise RuntimeError(f'Failed to load SAM: {e}')
    
    # Move to device
    sam = sam.to(args.device)
    
    if args.device == 'cuda':
        print(f'✅ SAM loaded on GPU: {torch.cuda.get_device_name(0)}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('⚠️  SAM loaded on CPU (slow)')
    
    # Create mask generator with performance-tuned settings
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    
    # ========================================================================
    # Process images
    # ========================================================================
    
    print(f'\n{"=" * 70}')
    print('Starting image processing')
    print(f'{"=" * 70}\n')
    
    processed = 0
    skipped_oom = 0
    skipped_no_mask = 0
    skipped_read_error = 0
    
    for img_name in tqdm(sampled, desc='Processing images'):
        src = DATASET_DIR / img_name
        dst_img = OUTPUT_DIR / 'images' / img_name
        
        # Read image
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            tqdm.write(f'⚠️  Failed to read {img_name}; skipping')
            skipped_read_error += 1
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        resized = img_rgb
        h, w = orig_h, orig_w
        
        # Downscale if needed to save GPU memory
        if args.max_image_size is not None:
            long_side = max(h, w)
            if long_side > args.max_image_size:
                scale = args.max_image_size / float(long_side)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = resized.shape[:2]
        
        # Generate masks with GPU acceleration
        try:
            with torch.no_grad():
                masks = mask_generator.generate(resized)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cuda' in msg:
                skipped_oom += 1
                tqdm.write(
                    f'\n⚠️  CUDA OOM on {img_name} (skipped {skipped_oom} total). '
                    f'Try reducing --max-image-size or --points-per-side'
                )
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                tqdm.write(f'⚠️  Failed to generate masks for {img_name}: {e}')
                continue
        
        if not masks:
            skipped_no_mask += 1
            continue
        
        # Sort masks by area descending
        masks_sorted = sorted(masks, key=lambda m: m.get('area', 0), reverse=True)
        
        # Consider top N largest masks and pick the one closest to image center
        candidates = masks_sorted[:args.top_n_masks]
        
        # Build list of (distance_to_center, mask, bbox) tuples
        candidate_distances = []
        for mask in candidates:
            bbox = mask.get('bbox')
            if bbox is None:
                seg = mask.get('segmentation')
                ys, xs = np.where(seg)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
            else:
                x_min, y_min, bw, bh = bbox
                x_max = x_min + bw
                y_max = y_min + bh
            
            # Compute distance to image center (on resized coords)
            dist = distance_to_image_center([x_min, y_min, x_max, y_max], w, h)
            candidate_distances.append((dist, mask, [x_min, y_min, x_max, y_max]))
        
        if not candidate_distances:
            skipped_no_mask += 1
            continue
        
        # Pick the mask with minimum distance to center
        candidate_distances.sort(key=lambda x: x[0])
        best_dist, best_mask, best_bbox_resized = candidate_distances[0]
        
        # Check area threshold
        if best_mask.get('area', 0) < args.area_threshold * (h * w):
            skipped_no_mask += 1
            continue
        
        x_min, y_min, x_max, y_max = best_bbox_resized
        
        # Scale back to original image coords if resized
        if (h, w) != (orig_h, orig_w):
            scale_x = orig_w / float(w)
            scale_y = orig_h / float(h)
            x_min *= scale_x
            x_max *= scale_x
            y_min *= scale_y
            y_max *= scale_y
        
        # Convert to YOLO format
        yolo_box = xyxy_to_yolo([x_min, y_min, x_max, y_max], orig_w, orig_h)
        
        # Get class ID
        cats = image_categories.get(img_name)
        label_name = cats[0] if isinstance(cats, list) else cats
        if label_name not in species_to_id:
            continue
        class_id = species_to_id[label_name]
        
        # Save image and label
        shutil.copy2(src, dst_img)
        label_path = OUTPUT_DIR / 'labels' / (img_name.rsplit('.', 1)[0] + '.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(f"{class_id} {' '.join([f'{v:.6f}' for v in yolo_box])}\n")
        
        # Save visualization with bbox and center crosshair
        vis = img_rgb.copy()
        # Draw selected bbox in green
        cv2.rectangle(vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
        # Draw crosshair at image center in blue
        center_x = int(orig_w / 2)
        center_y = int(orig_h / 2)
        cv2.line(vis, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(vis, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / 'vis' / img_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        processed += 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print(f'\n{"=" * 70}')
    print('Processing complete!')
    print(f'{"=" * 70}')
    print(f'✅ Successfully processed: {processed} images')
    print(f'⚠️  Skipped (OOM): {skipped_oom} images')
    print(f'⚠️  Skipped (no valid mask): {skipped_no_mask} images')
    print(f'⚠️  Skipped (read error): {skipped_read_error} images')
    print(f'\nOutput directory: {OUTPUT_DIR}')
    print(f'  - images/: {len(list((OUTPUT_DIR / "images").glob("*")))} files')
    print(f'  - labels/: {len(list((OUTPUT_DIR / "labels").glob("*")))} files')
    print(f'  - vis/: {len(list((OUTPUT_DIR / "vis").glob("*")))} files')
    print(f'  - classes.txt: {len(species_list)} classes')
    
    if args.device == 'cuda':
        print(f'\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
    
    print(f'{"=" * 70}\n')


if __name__ == '__main__':
    main()

