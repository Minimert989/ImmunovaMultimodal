import os
import json
import torch
from torch.utils.data import Dataset
import h5py

class TILPatientDataset(Dataset):
    def __init__(self, data_dir, mapping_json, label_type_filter=None, preload_limit=None, h5_paths=None):
        self.data_list = []

        if h5_paths:
            for h5_path in h5_paths:
                try:
                    with h5py.File(h5_path, 'r') as f:
                        images = f["images"][:]
                        labels = f["labels"][:]
                        if preload_limit:
                            images = images[:preload_limit]
                            labels = labels[:preload_limit]
                        for img, lbl in zip(images, labels):
                            self.data_list.append((torch.tensor(img), torch.tensor(lbl)))
                except Exception as e:
                    print(f"Failed to load HDF5 {h5_path}: {e}")
            return

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
