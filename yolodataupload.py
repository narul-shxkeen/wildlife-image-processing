# Create yolomodelimages folder with 1200 random images from 7 classes
import json
import random
from pathlib import Path
import shutil
import tqdm

# Configuration
DATASET_DIR = Path('dataset')
LABELS_JSON = Path('image_categories_cleaned.json')
OUTPUT_FOLDER = Path('yolomodelimages')
NUM_SAMPLES = 1200
SEED = 42

# The 7 species classes
ALLOWED_SPECIES = {
    "Common leopard",
    "Himalayan goral",
    "Rhesus macaque",
    "Himalayan gray langur",
    "Himalayan tahr",
    "Yellow-throated marten",
    "Leopard cat"
}

print(f'Loading labels from {LABELS_JSON}...')
with open(LABELS_JSON, 'r') as f:
    labels_data = json.load(f)

# Filter images that contain at least one of the allowed species
filtered_images = []
species_counts = {sp: 0 for sp in ALLOWED_SPECIES}

for img_name, categories in labels_data.items():
    # Skip if categories is None or empty
    if not categories:
        continue
    
    # Check if this image contains any of our 7 species
    img_species = set(categories) & ALLOWED_SPECIES
    if img_species:
        img_path = DATASET_DIR / img_name
        if img_path.exists():
            filtered_images.append(img_name)
            # Count species occurrences
            for sp in img_species:
                species_counts[sp] += 1

print(f'\n✅ Found {len(filtered_images)} images with the 7 target species')
print('\nSpecies distribution:')
for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {species}: {count} images')

# Sample random images
random.seed(SEED)
if len(filtered_images) < NUM_SAMPLES:
    print(f'\n⚠️ Warning: Only {len(filtered_images)} images available, requested {NUM_SAMPLES}')
    sampled_images = filtered_images
else:
    sampled_images = random.sample(filtered_images, NUM_SAMPLES)
    print(f'\n✅ Randomly selected {len(sampled_images)} images')

# Create output folder and copy images
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f'\nCopying images to {OUTPUT_FOLDER}/...')

copied_count = 0
failed_count = 0
for img_name in tqdm(sampled_images, desc='Copying images'):
    src_path = DATASET_DIR / img_name
    dst_path = OUTPUT_FOLDER / img_name
    
    try:
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    except Exception as e:
        print(f'Failed to copy {img_name}: {e}')
        failed_count += 1

print(f'\n✅ Successfully copied {copied_count} images to {OUTPUT_FOLDER}/')
if failed_count > 0:
    print(f'⚠️ Failed to copy {failed_count} images')

print(f'\nFolder ready for upload to HPC server!')
print(f'Total size: {sum(f.stat().st_size for f in OUTPUT_FOLDER.glob("*") if f.is_file()) / (1024**2):.2f} MB')