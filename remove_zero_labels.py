#!/usr/bin/env python3
"""
Script to remove all images labeled as 0 from threeClass.json.
This removes incorrectly labeled images from the training data.
"""

import json
import os
from pathlib import Path

def remove_zero_labels(input_file='threeClass.json', output_file='threeClass.json', backup=True):
    """
    Remove all entries with label 0 from the JSON file.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file (can be same as input)
        backup: If True, creates a backup of the original file
    """
    input_path = Path(input_file)
    
    # Check if file exists
    if not input_path.exists():
        print(f"Error: File '{input_file}' not found!")
        return
    
    # Load the JSON data
    print(f"Loading data from '{input_file}'...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data)
    print(f"Original dataset contains {original_count} images")
    
    # Count labels before filtering
    label_counts = {}
    for label in data.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")
    
    # Create backup if requested
    if backup and input_file == output_file:
        backup_path = input_path.with_suffix('.json.backup')
        print(f"Creating backup at '{backup_path}'...")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    # Filter out entries with label 0
    print("Removing entries with label 0...")
    filtered_data = {image_name: label for image_name, label in data.items() if label != 0}
    
    removed_count = original_count - len(filtered_data)
    new_count = len(filtered_data)
    
    print(f"Removed {removed_count} images with label 0")
    print(f"New dataset contains {new_count} images")
    
    # Count labels after filtering
    new_label_counts = {}
    for label in filtered_data.values():
        new_label_counts[label] = new_label_counts.get(label, 0) + 1
    print(f"New label distribution: {new_label_counts}")
    
    # Save the filtered data
    output_path = Path(output_file)
    print(f"Saving filtered data to '{output_file}'...")
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    # Run the script with default parameters
    # This will:
    # 1. Create a backup file (threeClass.json.backup)
    # 2. Remove all entries with label 0
    # 3. Overwrite threeClass.json with the filtered data
    remove_zero_labels()
