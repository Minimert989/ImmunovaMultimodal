import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import pandas as pd
import glob
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

from dataset import PatientDataset
from model import ImmunovaMultimodalModel

# === Configuration ===
task = "response"  # options: 'til', 'response', 'survival'
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

# === Prediction and Saving ===
all_labels = []
all_probs = []
all_ids = []

with torch.no_grad():
    for batch in val_loader:
        ids = batch["id"]
        wsi_feat = batch["wsi_feat"]
        if wsi_feat is not None:
            # wsi_feat: (B, N, 1024)
            wsi_feat = wsi_feat.to(device)

        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)

        response_label = batch["response_label"].to(device)
        til_label = batch["til_label"].to(device)

        til_pred, resp_pred, survival_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        if task == "response":
            mask = (response_label != -1)
            prob = torch.sigmoid(resp_pred[mask]).squeeze().cpu().numpy()
            label = response_label[mask].cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(label.tolist())
            all_ids.extend([i for i, m in zip(ids, mask) if m.item()])
        elif task == "til":
            mask = (til_label.sum(dim=1) >= 0)
            prob = torch.sigmoid(til_pred[mask]).cpu().numpy()
            label = til_label[mask].cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(label.tolist())
            all_ids.extend([i for i, m in zip(ids, mask) if m.item()])
        elif task == "survival":
            survival_time = batch["survival_time"].to(device)
            mask = (survival_time != -1)
            prob = survival_pred[mask].squeeze().cpu().numpy()
            label = survival_time[mask].cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(label.tolist())
            all_ids.extend([i for i, m in zip(ids, mask) if m.item()])

# === CSV Saving ===
df = pd.DataFrame({"id": all_ids, "label": all_labels, "prob": all_probs})
df.to_csv(f"predictions_{task}.csv", index=False)
print(f"[PREDICT] Saved predictions_{task}.csv")

# === Visualization ===
if task == "response":
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall, precision)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC={auc_pr:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.savefig("pr_curve.png")

    print("[PREDICT] Saved roc_curve.png and pr_curve.png")