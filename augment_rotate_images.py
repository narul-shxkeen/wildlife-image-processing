#!/usr/bin/env python3
"""
Script to augment image data by rotating images by 5 degrees.
This creates rotated versions of images while maintaining original dimensions.
The rotated images are saved with a '_rotated5' suffix to distinguish them from originals.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import random

def rotate_image(image_path, angle=5, save_path=None):
    """
    Rotate an image by a specified angle while maintaining original dimensions.
    
    Args:
        image_path: Path to the input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        save_path: Path to save the rotated image. If None, uses original name with suffix.
    
    Returns:
        Path to the saved rotated image
    """
    # Load the image
    img = Image.open(image_path)
    
    # Get original dimensions
    width, height = img.size
    
    # Rotate the image with expansion to avoid cropping
    # expand=False keeps the original dimensions, filling with black pixels where needed
    rotated = img.rotate(angle, expand=False, fillcolor=(0, 0, 0), resample=Image.BICUBIC)
    
    # Ensure the dimensions match the original
    if rotated.size != (width, height):
        # Center crop to original dimensions
        left = (rotated.width - width) // 2
        top = (rotated.height - height) // 2
        right = left + width
        bottom = top + height
        rotated = rotated.crop((left, top, right, bottom))
    
    # Determine save path,
    if save_path is None:
        path_obj = Path(image_path)
        save_path = path_obj.parent / f"{path_obj.stem}_rotated{angle}{path_obj.suffix}"
    
    # Save the rotated image
    rotated.save(save_path, quality=95)
    
    return save_path

def augment_folder(folder_path, num_rotations=25, extensions=('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'), seed=None):
    """
    Rotate all images in a folder by random angles multiple times.
    
    Args:
        folder_path: Path to the folder containing images
        num_rotations: Number of random rotations to create per image
        extensions: Tuple of valid image file extensions
        seed: Random seed for reproducibility (optional)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Get all image files in the folder (excluding already rotated images)
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix in extensions and '_rotated' not in f.stem]
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} original images to augment")
    print(f"Will create {num_rotations} random rotations per image")
    print(f"Total augmented images to create: {len(image_files) * num_rotations}")
    print("-" * 60)
    
    total_successful = 0
    total_failed = 0
    
    # Process each rotation iteration
    for rotation_num in range(1, num_rotations + 1):
        print(f"\n[Rotation Set {rotation_num}/{num_rotations}]")
        set_successful = 0
        set_failed = 0
        
        for idx, image_file in enumerate(image_files, 1):
            try:
                # Generate random angle between 0 and 360
                angle = random.randint(0, 360)
                
                # Create custom save path with rotation number
                path_obj = Path(image_file)
                save_path = path_obj.parent / f"{path_obj.stem}_rotated{angle}_v{rotation_num}{path_obj.suffix}"
                
                # Rotate and save
                rotate_image(image_file, angle=angle, save_path=save_path)
                
                if idx % 20 == 0 or idx == len(image_files):  # Print every 20th image or last one
                    print(f"  [{idx}/{len(image_files)}] Processed with angle {angle}°")
                
                set_successful += 1
                total_successful += 1
                
            except Exception as e:
                print(f"  [{idx}/{len(image_files)}] ✗ Failed: {image_file.name} - Error: {e}")
                set_failed += 1
                total_failed += 1
        
        print(f"  ✓ Completed rotation set {rotation_num}: {set_successful} successful, {set_failed} failed")
    
    print("-" * 60)
    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Original images: {len(image_files)}")
    print(f"Rotations per image: {num_rotations}")
    print(f"Successfully created: {total_successful} augmented images")
    print(f"Failed: {total_failed} images")
    print(f"\nFinal dataset size: {len(image_files)} (original) + {total_successful} (augmented) = {len(image_files) + total_successful} total images")
    print(f"Dataset increased by: {(total_successful / len(image_files) * 100):.1f}%")

if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "emptyImages"
    NUM_ROTATIONS = 25  # Number of random rotations per image
    RANDOM_SEED = None  # Set to a number for reproducible results, or None for truly random
    
    print("=" * 60)
    print("IMAGE AUGMENTATION - RANDOM ROTATIONS")
    print("=" * 60)
    print(f"Folder: {FOLDER_PATH}")
    print(f"Rotation angle range: 0° to 360° (random)")
    print(f"Rotations per image: {NUM_ROTATIONS}")
    print(f"Output: Rotated images will be saved in the same folder")
    print(f"Naming: Original_name_rotated[angle]_v[version].extension")
    print(f"Example: image.jpg -> image_rotated45_v1.jpg, image_rotated287_v2.jpg, etc.")
    print("=" * 60)
    print()
    
    # Run the augmentation
    augment_folder(FOLDER_PATH, num_rotations=NUM_ROTATIONS, seed=RANDOM_SEED)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
