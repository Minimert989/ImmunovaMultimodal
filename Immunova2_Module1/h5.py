import os
import torch
import h5py
from tqdm import tqdm

# 🔧 Hardcoded directory containing .pt files (no subfolders)
PT_DIR = "uvm"

def convert_pt_to_single_h5(pt_dir):
    pt_dir = os.path.abspath(pt_dir)
    folder_name = os.path.basename(pt_dir)
    root_dir = os.path.dirname(pt_dir)
    output_dir = os.path.join(root_dir, f"{folder_name}-h5")
    os.makedirs(output_dir, exist_ok=True)

    pt_files = [
        f for f in os.listdir(pt_dir)
        if f.endswith(".pt") and not f.startswith("._")
    ]

    all_images = []
    all_labels = []

    for filename in tqdm(pt_files, desc="Converting .pt files"):
        pt_path = os.path.join(pt_dir, filename)
        try:
            data = torch.load(pt_path, map_location="cpu")
            if not isinstance(data, dict):
                print(f"✖ Skipped {filename}: Not a dict")
                continue
            if "images" not in data or "labels" not in data:
                print(f"✖ Skipped {filename}: Missing 'images' or 'labels'")
                continue

            all_images.append(data["images"])
            all_labels.append(data["labels"])
        except Exception as e:
            print(f"✖ Failed to convert {filename}: {e}")

    if all_images and all_labels:
        combined_images = torch.cat(all_images, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)

        h5_path = os.path.join(output_dir, "combined.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("images", data=combined_images.numpy())
            f.create_dataset("labels", data=combined_labels.numpy())
        print(f"✅ Combined HDF5 saved: {h5_path}")
    else:
        print("❌ No valid data found for conversion.")

if __name__ == "__main__":
    convert_pt_to_single_h5(PT_DIR)
