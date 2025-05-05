import os
import pandas as pd
import shutil

# Define the classes we want to keep
CLASSES_TO_KEEP = ["Angora", "British Shorthair", "Persian"]


def clean_dataset(dataset_dir):
    # Process each split (train, test, valid)
    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(dataset_dir, split)

        # Create output directories if they don't exist
        output_dir = os.path.join(dataset_dir, f"{split}_cleaned")
        os.makedirs(output_dir, exist_ok=True)

        # Read the classes CSV file
        csv_path = os.path.join(split_dir, "_classes.csv")
        df = pd.read_csv(csv_path)

        # Create a new dataframe with only the columns we want to keep
        columns_to_keep = ["filename"] + CLASSES_TO_KEEP
        new_df = df[columns_to_keep].copy()

        # Filter rows where at least one of the classes we want to keep has a value of 1
        mask = new_df[CLASSES_TO_KEEP].sum(axis=1) > 0
        new_df = new_df[mask]

        # Save the new CSV file
        new_csv_path = os.path.join(output_dir, "_classes.csv")
        new_df.to_csv(new_csv_path, index=False)

        print(f"Processed {split} CSV: {len(new_df)} images kept out of {len(df)}")

        # Copy the images that we're keeping to the new directory
        for filename in new_df["filename"]:
            src_path = os.path.join(split_dir, filename)
            dst_path = os.path.join(output_dir, filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: File not found: {src_path}")

        print(f"Copied images for {split} to {output_dir}")


if __name__ == "__main__":
    dataset_dir = "dataset"
    clean_dataset(dataset_dir)
    print("Dataset cleaning completed!")
