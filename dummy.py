import pandas as pd
import os
import numpy as np
import torch
import pickle

df = pd.read_csv("Immunova2_module2/ACC/TCGA_rnaseq_ACC_immune_markers_with_metadata.csv", index_col=0)
print(df.head())
print(df.dtypes)

# --- Generate realistic dummy WSI feature data for all cancer types ---
cancer_list = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COADREAD", "DLBC", "ESCA", "GBM", "GBMLGG", "HNSC",
    "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD",
    "PCPG", "PRAD", "SARC", "SKCM", "STAD", "STES", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
]

os.makedirs("wsi_feature", exist_ok=True)

for cancer in cancer_list:
    dummy_wsi_features = {}
    for i in range(30):
        case_id = f"TCGA-DUMMY-{cancer}-{1000+i}"
        num_patches = np.random.randint(30, 51)
        tensor = torch.randn(num_patches, 512)
        dummy_wsi_features[case_id] = tensor

    out_path = f"wsi_feature/wsi_features_{cancer}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dummy_wsi_features, f)
    print(f"✅ Saved dummy WSI features: {out_path}")

# --- Generate dummy WSI features using label_dict keys ---
import glob

# Load label dicts and collect keys
label_keys = set()
for path in glob.glob("label_feature/label_dict_*.pkl"):
    with open(path, "rb") as f:
        label_dict = pickle.load(f)
        label_keys.update(label_dict.keys())

# Create matching WSI dummy features
wsi_from_label = {}
for pid in label_keys:
    num_patches = np.random.randint(30, 51)
    wsi_from_label[pid] = torch.randn(num_patches, 512)

# Save as unified WSI feature file
os.makedirs("wsi_feature", exist_ok=True)
with open("wsi_feature/wsi_features_DUMMY_FROM_LABEL.pkl", "wb") as f:
    pickle.dump(wsi_from_label, f)

print(f"✅ Saved WSI features based on label_dict keys: {len(wsi_from_label)} samples")
