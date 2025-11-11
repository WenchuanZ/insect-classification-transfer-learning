"""
Script to reorganize YOLO format dataset into PyTorch ImageFolder format.
Converts from images/ + labels/ structure to class-based subdirectories.
"""

import os
import shutil
from pathlib import Path

# Class names from data.yaml
class_names = [
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
]

def reorganize_split(split_name, base_dir="/home/ian/Projects/datas"):
    """Reorganize a single split (train/valid/test) into class directories."""
    
    images_dir = Path(base_dir) / split_name / "images"
    labels_dir = Path(base_dir) / split_name / "labels"
    output_dir = Path(base_dir) / f"{split_name}_organized"
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist")
        return
    
    # Create class directories
    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Processing {len(image_files)} images in {split_name}...")
    
    images_processed = 0
    images_skipped = 0
    
    for img_path in image_files:
        # Find corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if label_path.exists():
            # Read the first line to get class ID
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    class_name = class_names[class_id]
                    
                    # Copy image to class directory
                    dest_path = output_dir / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
                    images_processed += 1
                else:
                    images_skipped += 1
        else:
            # No label file - try to infer from filename
            filename_lower = img_path.name.lower()
            matched = False
            for class_name in class_names:
                if class_name.lower() in filename_lower:
                    dest_path = output_dir / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
                    images_processed += 1
                    matched = True
                    break
            if not matched:
                images_skipped += 1
    
    print(f"  Processed: {images_processed}, Skipped: {images_skipped}")
    
    # Print class distribution
    print(f"\nClass distribution for {split_name}:")
    for class_name in class_names:
        class_dir = output_dir / class_name
        count = len(list(class_dir.glob("*")))
        print(f"  {class_name}: {count}")

if __name__ == "__main__":
    print("Reorganizing dataset for PyTorch ImageFolder format...\n")
    
    # Reorganize each split
    for split in ["train", "valid", "test"]:
        reorganize_split(split)
        print()
    
    print("Dataset reorganization complete!")
    print("\nOrganized directories:")
    print("  - /home/ian/Projects/datas/train_organized/")
    print("  - /home/ian/Projects/datas/valid_organized/")
    print("  - /home/ian/Projects/datas/test_organized/")


