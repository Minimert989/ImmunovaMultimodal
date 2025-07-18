import sys
import torch
from torch.utils.data import DataLoader
from model import ImmunovaMultimodalModel
from dataset import PatientDataset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import glob
import logging
import time

# --- 1. LOGGING SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('training.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# --- 2. CONFIGURATION ---
TASK = "survival"  # options: 'til', 'response', 'survival'
INPUT_DIMS = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

logger.info("--- Starting Training Script ---")
logger.info(f"Configuration: Task={TASK}, Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")

# --- 3. DATA LOADING ---
logger.info("Loading features...")

# Load all WSI features
wsi_features = {}
for path in glob.glob("wsi_feature/wsi_features_*.pkl"):
    with open(path, "rb") as f:
        wsi_features.update(pickle.load(f))
logger.info(f"✅ Loaded WSI features: {len(wsi_features)} samples")

# Load all label dicts
label_dict = {}
for path in glob.glob("label_feature/label_dict_*.pkl"):
    with open(path, "rb") as f:
        label_dict.update(pickle.load(f))
logger.info(f"✅ Loaded label dicts: {len(label_dict)} samples")

# Diagnostic: Check how many patients have a valid 'response' label
valid_response_labels = sum(1 for pid, data in label_dict.items() if 'response' in data and data['response'].item() != -1.0)
logger.info(f"Diagnostic: Number of patients with valid 'response' labels: {valid_response_labels}")

# Load omics dict
with open("omics_feature/omics_dict.pkl", "rb") as f:
    omics_dict = pickle.load(f)
logger.info(f"✅ Loaded omics types: {list(omics_dict.keys())}")
for k in omics_dict:
    logger.info(f"   - {k}: {len(omics_dict[k])} samples")

# --- 4. DATA SPLITTING ---
logger.info("Splitting data into training and validation sets...")
common_ids = list(set(wsi_features.keys()) & set(label_dict.keys()))
train_ids, val_ids = train_test_split(common_ids, test_size=0.2, random_state=42)
logger.info(f"✅ Total common patients: {len(common_ids)}")
logger.info(f"📊 Train/Val split → Train: {len(train_ids)}, Val: {len(val_ids)}")

with open("val_ids.txt", "w") as f:
    f.writelines([pid + "\n" for pid in val_ids])
logger.info("Saved validation IDs to val_ids.txt")

# --- 5. DATASET & DATALOADER ---
logger.info("Creating Datasets and DataLoaders...")
train_dataset = PatientDataset(train_ids, wsi_features, omics_dict, label_dict, INPUT_DIMS)
val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, INPUT_DIMS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
logger.info("DataLoaders are ready.")

# --- 6. MODEL & OPTIMIZER ---
logger.info("Initializing model and optimizer...")
model = ImmunovaMultimodalModel(input_dims=(*INPUT_DIMS.values(),))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model initialized and moved to device: {device}")

# --- 7. TRAINING LOOP ---
logger.info("--- Starting Model Training ---")
overall_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move tensors to device
        wsi_feat = batch["wsi_feat"].to(device) if batch["wsi_feat"] is not None else None
        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)

        # Labels
        til_label = batch["til_label"].to(device)
        response_label = batch["response_label"].to(device)
        survival_time = batch["survival_time"].to(device)

        til_pred, resp_pred, survival_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        loss = 0
        if TASK == "til":
            mask = (til_label.sum(dim=1) >= 0)
            if mask.any(): loss = F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask])
        elif TASK == "response":
            mask = (response_label != -1).squeeze() # Ensure mask is 1D
            logger.debug(f"Train Loop - resp_pred shape: {resp_pred.shape}, response_label shape: {response_label.shape}, mask shape: {mask.shape}")
            if mask.any():
                valid_indices = mask.nonzero(as_tuple=True)[0]
                loss = F.binary_cross_entropy_with_logits(resp_pred.squeeze()[valid_indices], response_label.squeeze()[valid_indices])
        elif TASK == "survival":
            mask = (survival_time != -1).squeeze() # Ensure mask is 1D
            logger.debug(f"Train Loop - survival_pred shape: {survival_pred.shape}, survival_time shape: {survival_time.shape}, mask shape: {mask.shape}")
            if mask.any():
                valid_indices = mask.nonzero(as_tuple=True)[0]
                loss = F.mse_loss(survival_pred.squeeze()[valid_indices], survival_time.squeeze()[valid_indices])
        else:
            logger.error(f"Invalid task specified: {TASK}")
            raise ValueError("Invalid task")

        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            wsi_feat = batch["wsi_feat"].to(device) if batch["wsi_feat"] is not None else None
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

            val_loss = 0
            if TASK == "til":
                mask = (til_label.sum(dim=1) >= 0)
                if mask.any(): val_loss = F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask])
            elif TASK == "response":
                mask = (response_label != -1)
                logger.debug(f"Val Loop - resp_pred shape: {resp_pred.shape}, response_label shape: {response_label.shape}, mask shape: {mask.shape}")
                if mask.any():
                    valid_indices = mask.nonzero(as_tuple=True)[0]
                    val_loss = F.binary_cross_entropy_with_logits(resp_pred.squeeze()[valid_indices], response_label.squeeze()[valid_indices])
            elif TASK == "survival":
                mask = (survival_time != -1)
                logger.debug(f"Val Loop - survival_pred shape: {survival_pred.shape}, survival_time shape: {survival_time.shape}, mask shape: {mask.shape}")
                if mask.any():
                    valid_indices = mask.nonzero(as_tuple=True)[0]
                    val_loss = F.mse_loss(survival_pred.squeeze()[valid_indices], survival_time.squeeze()[valid_indices])

            if isinstance(val_loss, torch.Tensor):
                total_val_loss += val_loss.item()

    epoch_end_time = time.time()
    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {epoch_end_time - epoch_start_time:.2f}s")

# --- 8. SAVE MODEL ---
logger.info("--- Finished Model Training ---")
overall_end_time = time.time()
logger.info(f"Total training time: {overall_end_time - overall_start_time:.2f} seconds")

try:
    torch.save(model.state_dict(), "model.pth")
    logger.info("✅ Trained model saved successfully to model.pth")
except Exception as e:
    logger.error(f"Failed to save model. Reason: {e}", exc_info=True)
