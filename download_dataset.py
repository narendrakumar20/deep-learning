"""
Download the Chest X-Ray (Pneumonia) dataset from Kaggle using kagglehub.
The dataset will be symlinked/copied into the local 'dataset/' directory.
"""

import os
import shutil
import kagglehub

def main():
    print("=" * 60)
    print("  Downloading Chest X-Ray (Pneumonia) Dataset from Kaggle")
    print("=" * 60)

    # Download the dataset via kagglehub
    dataset_path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia"
    )
    print(f"\nDataset downloaded to: {dataset_path}")

    # The dataset structure from Kaggle:
    # <download_path>/chest_xray/chest_xray/{train, val, test}/
    # OR <download_path>/chest_xray/{train, val, test}/
    # We need to locate the actual data root that contains train/val/test

    source = None
    for candidate in [
        os.path.join(dataset_path, "chest_xray", "chest_xray"),
        os.path.join(dataset_path, "chest_xray"),
        dataset_path,
    ]:
        if os.path.isdir(os.path.join(candidate, "train")):
            source = candidate
            break

    if source is None:
        print("\n[ERROR] Could not locate train/val/test directories in the download.")
        print(f"Please check the contents of: {dataset_path}")
        return

    # Create a symlink or copy into local 'dataset/' directory
    local_dataset = os.path.join(os.path.dirname(__file__), "dataset")

    if os.path.exists(local_dataset):
        print(f"\n'dataset/' directory already exists. Skipping copy.")
    else:
        print(f"\nCopying dataset to {local_dataset} ...")
        shutil.copytree(source, local_dataset)
        print("Done!")

    # Print dataset statistics
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(local_dataset, split)
        if os.path.isdir(split_dir):
            for label in os.listdir(split_dir):
                label_dir = os.path.join(split_dir, label)
                if os.path.isdir(label_dir):
                    count = len(os.listdir(label_dir))
                    print(f"  {split}/{label}: {count} images")

    print("\n✅ Dataset is ready in 'dataset/' directory.")


if __name__ == "__main__":
    main()
