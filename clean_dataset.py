import os
import pandas as pd
import shutil

# Define the classes we want to keep
CLASSES_TO_KEEP = ["Angora", "British Shorthair", "Persian"]


def clean_dataset(dataset_dir, output_dir="dataset_restructured"):
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each split (train, test, valid)
    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(dataset_dir, split)

        # Create output split directory
        output_split_dir = os.path.join(output_dir, split.replace("valid", "val"))

        # Create class directories for this split
        for class_name in CLASSES_TO_KEEP:
            class_dir = os.path.join(
                output_split_dir, class_name.lower().replace(" ", "_")
            )
            os.makedirs(class_dir, exist_ok=True)

        # Read the classes CSV file
        csv_path = os.path.join(split_dir, "_classes.csv")
        df = pd.read_csv(csv_path)

        # Filter rows where at least one of the classes we want to keep has a value of 1
        columns_to_keep = ["filename"] + CLASSES_TO_KEEP
        filtered_df = df[columns_to_keep].copy()

        # Process each image and copy to appropriate class folder
        for _, row in filtered_df.iterrows():
            # Find which class this image belongs to
            for class_name in CLASSES_TO_KEEP:
                if row[class_name] == 1:
                    # Source path
                    src_path = os.path.join(split_dir, row["filename"])

                    # Destination path (in class-specific folder)
                    dst_folder = os.path.join(
                        output_split_dir, class_name.lower().replace(" ", "_")
                    )
                    dst_path = os.path.join(dst_folder, row["filename"])

                    # Copy the image if it exists
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                    else:
                        print(f"Warning: File not found: {src_path}")

        print(f"Processed {split} split")

    print(f"Dataset restructured to {output_dir}")


if __name__ == "__main__":
    dataset_dir = "dataset"
    clean_dataset(dataset_dir)
    print("Dataset restructuring completed!")
