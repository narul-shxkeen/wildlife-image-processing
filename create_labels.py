#!/usr/bin/env python3
"""
Script to create binary labels for wildlife image classification.
Assigns label 0 to images with null categories (empty images)
and label 1 to images with valid categories (wildlife present).
"""

import json
import sys
from pathlib import Path

def create_binary_labels(input_file: str, output_file: str = "labels.json") -> None:
    """
    Process the image categories JSON file and create binary labels.
    
    Args:
        input_file (str): Path to the input JSON file with image categories
        output_file (str): Path to the output labels JSON file
    """
    
    # Read the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            image_categories = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        sys.exit(1)
    
    # Create binary labels
    labels = {}
    empty_count = 0
    wildlife_count = 0
    
    for image_name, categories in image_categories.items():
        if categories is None:
            # No wildlife detected - assign label 0
            labels[image_name] = 0
            empty_count += 1
        else:
            # Wildlife detected - assign label 1
            labels[image_name] = 1
            wildlife_count += 1
    
    # Write the labels to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error: Could not write to '{output_file}': {e}")
        sys.exit(1)
    
    # Print summary statistics
    total_images = len(labels)
    print(f"Binary labels created successfully!")
    print(f"Output file: {output_file}")
    print(f"\nSummary:")
    print(f"Total images: {total_images:,}")
    print(f"Empty images (label 0): {empty_count:,} ({empty_count/total_images*100:.1f}%)")
    print(f"Wildlife images (label 1): {wildlife_count:,} ({wildlife_count/total_images*100:.1f}%)")
    
    return labels

def main():
    """Main function to run the script."""
    
    # Default input and output files
    input_file = "image_categories_cleaned.json"
    output_file = "labels.json"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found in current directory.")
        print("Please make sure 'image_categories_cleaned.json' exists in the current directory.")
        sys.exit(1)
    
    # Create binary labels
    create_binary_labels(input_file, output_file)

if __name__ == "__main__":
    main()