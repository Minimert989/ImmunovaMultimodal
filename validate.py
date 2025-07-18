

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import glob
from dataset import PatientDataset
from model import ImmunovaMultimodalModel

# === Configuration ===
task = "survival"  # options: 'til', 'response', 'survival'
input_dims = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Loading ===
# Load all WSI features
wsi_features = {}
for path in glob.glob("wsi_feature/wsi_features_*.pkl"):
    with open(path, "rb") as f:
        wsi_features.update(pickle.load(f))
print(f"Loaded WSI features: {len(wsi_features)} samples")

# Load omics dict
with open("omics_feature/omics_dict.pkl", "rb") as f:
    omics_dict = pickle.load(f)
print(f"Loaded omics types: {list(omics_dict.keys())}")

# Load all label dicts
label_dict = {}
for path in glob.glob("label_feature/label_dict_*.pkl"):
    with open(path, "rb") as f:
        label_dict.update(pickle.load(f))
print(f"Loaded label dicts: {len(label_dict)} samples")

with open("val_ids.txt", "r") as f:
    val_ids = [line.strip() for line in f.readlines()]

val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, input_dims)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Model Loading ===
model = ImmunovaMultimodalModel(input_dims=(1000, 512, 256, 128))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# === Validation ===
total_loss = 0
with torch.no_grad():
    for batch in val_loader:
        wsi_feat = batch["wsi_feat"]
        if wsi_feat is not None:
            wsi_feat = wsi_feat.to(device)

        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)

        til_label = batch["til_label"].to(device)
        response_label = batch["response_label"].to(device)
        survival_time = batch["survival_time"].to(device)

        til_pred, resp_pred, survival_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        val_loss = torch.tensor(0.0).to(device)
        if task == "til":
            mask = (til_label.sum(dim=1) >= 0)
            if mask.any(): val_loss = F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask])
        elif task == "response":
            mask = (response_label != -1).squeeze()
            if mask.any(): val_loss = F.binary_cross_entropy_with_logits(resp_pred.squeeze()[mask], response_label.squeeze()[mask])
        elif task == "survival":
            mask = (survival_time != -1).squeeze()
            if mask.any(): val_loss = F.mse_loss(survival_pred.squeeze()[mask], survival_time.squeeze()[mask])
        else:
            raise ValueError("Unknown task")

        total_loss += val_loss.item()

print(f"[VALIDATION] {task} Loss: {total_loss:.4f}")