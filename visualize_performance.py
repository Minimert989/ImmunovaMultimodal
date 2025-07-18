import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from torch.utils.data import DataLoader

from dataset import PatientDataset
from model import ImmunovaMultimodalModel

import logging

# --- 1. LOGGING SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('visualize_performance.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# === Configuration ===
INPUT_DIMS = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Loading ===
wsi_features = {}
for path in glob.glob("wsi_feature/wsi_features_*.pkl"):
    with open(path, "rb") as f:
        wsi_features.update(pickle.load(f))

omics_dict = {}
with open("omics_feature/omics_dict.pkl", "rb") as f:
    omics_dict = pickle.load(f)

label_dict = {}
for path in glob.glob("label_feature/label_dict_*.pkl"):
    with open(path, "rb") as f:
        label_dict.update(pickle.load(f))

with open("val_ids.txt", "r") as f:
    val_ids = [line.strip() for line in f.readlines()]

val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, INPUT_DIMS)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Model Loading ===
model = ImmunovaMultimodalModel(input_dims=(*INPUT_DIMS.values(),))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# === Prediction and Visualization ===
all_actuals = []
all_predictions = []

with torch.no_grad():
    for batch in val_loader:
        wsi_feat = batch["wsi_feat"].to(device) if batch["wsi_feat"] is not None else None
        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)
        survival_time = batch["survival_time"].to(device)

        _, _, survival_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        mask = (survival_time != -1).squeeze()
        if mask.any():
            valid_indices = mask.nonzero(as_tuple=True)[0]
            
            # Convert predictions and actuals back from log1p scale
            actuals = torch.expm1(survival_time.squeeze()[valid_indices]).cpu().numpy()
            predictions = torch.expm1(survival_pred.squeeze()[valid_indices]).cpu().numpy()
            
            all_actuals.extend(actuals)
            all_predictions.extend(predictions)

logger.info(f"Actuals (expm1) - Min: {np.min(all_actuals):.2f}, Max: {np.max(all_actuals):.2f}, Mean: {np.mean(all_actuals):.2f}, Std: {np.std(all_actuals):.2f}")
logger.info(f"Predictions (expm1) - Min: {np.min(all_predictions):.2f}, Max: {np.max(all_predictions):.2f}, Mean: {np.mean(all_predictions):.2f}, Std: {np.std(all_predictions):.2f}")

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(all_actuals, all_predictions, alpha=0.6)
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], '--r', label='Perfect Prediction')

plt.xlabel("Actual Survival Time (days)")
plt.ylabel("Predicted Survival Time (days)")
plt.title("Predicted vs. Actual Survival Times")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("survival_prediction_plot.png")
print("✅ Survival prediction plot saved to survival_prediction_plot.png")
