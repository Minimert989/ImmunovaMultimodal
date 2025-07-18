import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
from model import TILBinaryCNN
import h5py
import numpy as np
import os
import matplotlib.gridspec as gridspec
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Configuration
model_path = "til_model.pth"
h5_files = [
    "acc-h5/combined.h5",
    "blca-h5/combined.h5",
    "brca-h5/combined.h5",
    "cesc-h5/combined.h5",
]
batch_size = 16
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = TILBinaryCNN(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

for h5_path in h5_files:
    label = os.path.basename(os.path.dirname(h5_path))  # ex: 'blca-h5'
    print(f"Evaluating {label}")

    with h5py.File(h5_path, "r") as f:
        images = torch.tensor(f["images"][:], dtype=torch.float32)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)

    test_dataset = TensorDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            preds_np = preds
            if preds_np.ndim == 0:
                preds_np = np.expand_dims(preds_np, axis=0)
            y_pred.extend(preds_np)
            y_true.extend(y.numpy())
            probs_np = probs.cpu().numpy()
            if probs_np.ndim == 0:
                probs_np = np.expand_dims(probs_np, axis=0)
            y_scores.extend(probs_np)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No TIL", "TIL"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({label})")
    plt.savefig(os.path.join(output_dir, f"performance_confmat_{label}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({label})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"performance_roc_{label}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({label})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"performance_pr_{label}.png"), dpi=300, bbox_inches="tight")
    plt.close()

labels = ["acc-h5", "blca-h5", "brca-h5", "cesc-h5"]
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(len(labels), 3, figure=fig)

for i, label in enumerate(labels):
    cm_img = plt.imread(os.path.join(output_dir, f"performance_confmat_{label}.png"))
    roc_img = plt.imread(os.path.join(output_dir, f"performance_roc_{label}.png"))
    pr_img = plt.imread(os.path.join(output_dir, f"performance_pr_{label}.png"))

    ax1 = fig.add_subplot(gs[i, 0])
    ax1.imshow(cm_img)
    ax1.axis("off")
    ax1.set_title(f"Confusion Matrix ({label})")

    ax2 = fig.add_subplot(gs[i, 1])
    ax2.imshow(roc_img)
    ax2.axis("off")
    ax2.set_title(f"ROC Curve ({label})")

    ax3 = fig.add_subplot(gs[i, 2])
    ax3.imshow(pr_img)
    ax3.axis("off")
    ax3.set_title(f"PR Curve ({label})")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_performance_summary.png"), dpi=300)
plt.close()

print("All individual and combined evaluation plots saved in the 'output' folder.")

# Combined Evaluation
print("\nEvaluating on all combined datasets...")
all_y_true, all_y_pred, all_y_scores = [], [], []

for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        images = torch.tensor(f["images"][:], dtype=torch.float32)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)

    test_dataset = TensorDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            if preds.ndim == 0:
                preds = np.expand_dims(preds, axis=0)
            all_y_pred.extend(preds)
            all_y_true.extend(y.numpy())
            probs_np = probs.cpu().numpy()
            if probs_np.ndim == 0:
                probs_np = np.expand_dims(probs_np, axis=0)
            all_y_scores.extend(probs_np)

# Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No TIL", "TIL"])
disp.plot(cmap="Purples", values_format="d")
plt.title("Confusion Matrix (All Combined)")
plt.savefig(os.path.join(output_dir, "combined_confmat.png"), dpi=300, bbox_inches="tight")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (All Combined)")
plt.legend()
plt.savefig(os.path.join(output_dir, "combined_roc.png"), dpi=300, bbox_inches="tight")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
plt.figure()
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (All Combined)")
plt.legend()
plt.savefig(os.path.join(output_dir, "combined_pr.png"), dpi=300, bbox_inches="tight")
plt.close()