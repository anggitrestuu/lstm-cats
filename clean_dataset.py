import os
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split

# Define the classes we want to keep (updated to match your folder structure)
CLASSES_TO_KEEP = ["domistik", "persian", "turkish"]


def clean_dataset(dataset_dir, output_dir="dataset_restructured", train_ratio=0.9):
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create output split directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create class directories for each split
    for class_name in CLASSES_TO_KEEP:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Process each class
    for class_name in CLASSES_TO_KEEP:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        # Get all image files in this class directory
        image_files = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Split into train and validation sets
        train_files, val_files = train_test_split(
            image_files, train_size=train_ratio, random_state=42
        )

        # Copy training images
        for filename in train_files:
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(train_dir, class_name, filename)
            shutil.copy2(src_path, dst_path)

        # Copy validation images
        for filename in val_files:
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(val_dir, class_name, filename)
            shutil.copy2(src_path, dst_path)

        print(
            f"Processed {class_name}: {len(train_files)} training, {len(val_files)} validation"
        )

    print(f"Dataset restructured to {output_dir}")


if __name__ == "__main__":
    dataset_dir = "dataset"
    clean_dataset(dataset_dir, train_ratio=0.9)
    print("Dataset restructuring completed!")
