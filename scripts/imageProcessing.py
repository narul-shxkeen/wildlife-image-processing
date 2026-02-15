#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Count the number of images in the T15 folder
t15_folder = "dataset"
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# Get all files in the T15 folder
files = os.listdir(t15_folder)

# Filter for image files (case insensitive)
image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]

# Count the images
image_count = len(image_files)

print(f"Number of images in {t15_folder} folder: {image_count}")
print(f"Total files in folder: {len(files)}")

# Optional: Show first few image names
if image_files:
    print(f"\nFirst 5 image files:")
    for img in image_files[:5]:
        print(f"  - {img}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")


# In[2]:


import os
import subprocess
import json
import re
import xml.etree.ElementTree as ET

def parse_categories_xml(categories_xml):
    """
    Parse the XML categories string and extract category names
    """
    if not categories_xml:
        return None

    try:
        # Parse the XML string
        root = ET.fromstring(categories_xml)

        # Extract all category text content
        categories = []

        def extract_category_text(element):
            # Get direct text content (category name)
            if element.text and element.text.strip():
                categories.append(element.text.strip())

            # Recursively process child elements
            for child in element:
                extract_category_text(child)

        extract_category_text(root)

        # Filter out empty strings and return unique categories
        categories = [cat for cat in categories if cat and cat != "Species"]
        return categories if categories else None

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

def extract_image_categories_cleaned():
    """
    Extract image names and their cleaned categories from T15 folder using exiftool
    Returns a dictionary with image names as keys and cleaned category lists as values
    """
    t15_folder = "dataset"
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    # Get all image files in the T15 folder
    files = os.listdir(t15_folder)
    image_files = [f for f in files if any(f.endswith(ext) for ext in image_extensions)]

    results = {}

    print(f"Processing {len(image_files)} images...")

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(t15_folder, image_file)

        try:
            # Run exiftool to get metadata for the specific image
            result = subprocess.run(
                ['exiftool', '-Categories', '-json', image_path],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the JSON output from exiftool
            metadata = json.loads(result.stdout)

            # Extract categories (exiftool returns a list, we want the first item)
            if metadata and len(metadata) > 0:
                categories_xml = metadata[0].get('Categories', None)

                # Parse the XML and extract clean category names
                clean_categories = parse_categories_xml(categories_xml)
                results[image_file] = clean_categories
            else:
                results[image_file] = None

        except subprocess.CalledProcessError as e:
            print(f"Error processing {image_file}: {e}")
            results[image_file] = None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {image_file}: {e}")
            results[image_file] = None

        # Progress indicator
        if i % 10 == 0:
            print(f"Processed {i}/{len(image_files)} images...")

    return results

# Extract and clean categories for all images
print("Extracting and cleaning image categories...")
clean_image_categories = extract_image_categories_cleaned()

# Output results in JSON format
print("\n" + "="*50)
print("CLEANED RESULTS IN JSON FORMAT:")
print("="*50)
print(json.dumps(clean_image_categories, indent=2, ensure_ascii=False))

# Save cleaned results to file
output_file = "image_categories_cleaned.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(clean_image_categories, f, indent=2, ensure_ascii=False)

print(f"\nCleaned results saved to: {output_file}")

# Summary statistics
total_images = len(clean_image_categories)
images_with_categories = sum(1 for v in clean_image_categories.values() if v is not None)
images_without_categories = total_images - images_with_categories

print(f"\nSUMMARY:")
print(f"Total images processed: {total_images}")
print(f"Images with categories: {images_with_categories}")
print(f"Images without categories: {images_without_categories}")

# Show unique species found
all_species = set()
for categories in clean_image_categories.values():
    if categories:
        all_species.update(categories)

print(f"\nUNIQUE SPECIES FOUND:")
for species in sorted(all_species):
    print(f"  - {species}")

print(f"\nTotal unique species: {len(all_species)}")

