#!/usr/bin/env python3
"""
Script to add all images from emptyImages folder to threeClass.json with label 0.
This is used to include the augmented empty images in the training dataset.
"""

import json
from pathlib import Path

def add_empty_images_to_labels(
    images_folder='emptyImages',
    labels_file='threeClass.json',
    label=0,
    backup=True,
    extensions=('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
):
    """
    Add all images from a folder to the labels JSON file with a specified label.
    
    Args:
        images_folder: Path to the folder containing images to add
        labels_file: Path to the JSON file containing image labels
        label: The label to assign to all images (default: 0)
        backup: If True, creates a backup of the original labels file
        extensions: Tuple of valid image file extensions
    """
    images_path = Path(images_folder)
    labels_path = Path(labels_file)
    
    # Check if images folder exists
    if not images_path.exists():
        print(f"Error: Folder '{images_folder}' not found!")
        return
    
    # Check if labels file exists
    if not labels_path.exists():
        print(f"Error: Labels file '{labels_file}' not found!")
        return
    
    # Load the existing labels
    print(f"Loading labels from '{labels_file}'...")
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    original_count = len(labels_data)
    print(f"Original labels file contains {original_count} images")
    
    # Count existing labels
    label_counts = {}
    for lbl in labels_data.values():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"Label distribution: {label_counts}")
    
    # Create backup if requested
    if backup:
        backup_path = labels_path.with_suffix('.json.backup2')
        print(f"Creating backup at '{backup_path}'...")
        with open(backup_path, 'w') as f:
            json.dump(labels_data, f, indent=4)
    
    # Get all image files from the folder
    image_files = [f.name for f in images_path.iterdir() 
                   if f.is_file() and f.suffix in extensions]
    
    if not image_files:
        print(f"No image files found in '{images_folder}'")
        return
    
    print(f"\nFound {len(image_files)} images in '{images_folder}'")
    
    # Add images to labels
    added_count = 0
    updated_count = 0
    
    for image_name in image_files:
        if image_name in labels_data:
            # Image already exists, update its label
            old_label = labels_data[image_name]
            if old_label != label:
                labels_data[image_name] = label
                updated_count += 1
        else:
            # New image, add it
            labels_data[image_name] = label
            added_count += 1
    
    print(f"\nAdded {added_count} new images with label {label}")
    print(f"Updated {updated_count} existing images to label {label}")
    
    new_count = len(labels_data)
    print(f"New labels file contains {new_count} images (was {original_count})")
    
    # Count labels after adding
    new_label_counts = {}
    for lbl in labels_data.values():
        new_label_counts[lbl] = new_label_counts.get(lbl, 0) + 1
    print(f"New label distribution: {new_label_counts}")
    
    # Save the updated labels
    print(f"\nSaving updated labels to '{labels_file}'...")
    with open(labels_path, 'w') as f:
        json.dump(labels_data, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    print("=" * 60)
    print("ADD EMPTY IMAGES TO LABELS")
    print("=" * 60)
    print("This script will:")
    print("1. Load threeClass.json")
    print("2. Find all images in emptyImages folder")
    print("3. Add them to the labels file with label 0")
    print("4. Create a backup before making changes")
    print("=" * 60)
    print()
    
    # Run the script
    add_empty_images_to_labels(
        images_folder='emptyImages',
        labels_file='threeClass.json',
        label=0,
        backup=True
    )
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
