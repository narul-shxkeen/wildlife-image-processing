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

# All allowed image extensions
img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

count = 0
for img_path in source_dir.rglob("*"):
    if img_path.is_file() and img_path.suffix.lower() in img_exts:
        # >>> Skip hidden Apple “._” files <<<
        if img_path.name.startswith("._"):
            continue  

        parent = img_path.parent.name  # e.g. 't15'
        new_name = f"{parent}{img_path.name}"  # t15image1.jpg
        dest_path = dest_dir / new_name

        # If same name already exists, append a number
        i = 1
        while dest_path.exists():
            stem = dest_path.stem
            ext = dest_path.suffix
            dest_path = dest_dir / f"{stem}_{i}{ext}"
            i += 1

        shutil.copy2(img_path, dest_path)
        count += 1

print(f"Copied {count} images to {dest_dir}")
