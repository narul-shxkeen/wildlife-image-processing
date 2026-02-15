#!/usr/bin/env python3
import sys
from pathlib import Path
import shutil

if len(sys.argv) != 3:
    print("Usage: python migrate_images.py <source_images_folder> <destination_folder>")
    sys.exit(1)

source_dir = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])

if not source_dir.exists():
    print(f"Source folder {source_dir} does not exist.")
    sys.exit(1)

dest_dir.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

count = 0
for img_path in source_dir.rglob("*"):
    if img_path.is_file() and img_path.suffix.lower() in img_exts:
        # Skip hidden Apple “._” files
        if img_path.name.startswith("._"):
            continue  

        # Get relative path from the source directory
        rel_path = img_path.relative_to(source_dir).with_suffix('')  # drop extension for now
        # Replace path separators with underscores for uniqueness
        rel_str = str(rel_path).replace('/', '_').replace('\\', '_')

        new_name = f"{rel_str}{img_path.suffix}"  # e.g., subdir1_subdir2_image1.jpg
        dest_path = dest_dir / new_name

        # If same name exists, append a number
        i = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{rel_str}_{i}{img_path.suffix}"
            i += 1

        shutil.copy2(img_path, dest_path)
        count += 1

print(f"✅ Copied {count} images to {dest_dir}")
