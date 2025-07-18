import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import models
from tqdm import tqdm

from til_dataset import TILPatientDataset
from model import TILBinaryCNN  # Assumes a custom CNN model for binary TIL classification
from grad_cam import get_gradcam_visualization  # GradCAM integration utility

# Configuration
DATA_DIR = "."
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

os.makedirs("gradcam_viz", exist_ok=True)

H5_PATHS = [
    os.path.join(DATA_DIR, "acc-h5/combined.h5"),
    os.path.join(DATA_DIR, "blca-h5/combined.h5"),
    os.path.join(DATA_DIR, "brca-h5/combined.h5"),
    os.path.join(DATA_DIR, "cesc-h5/combined.h5"),
    os.path.join(DATA_DIR, "coad-h5/combined.h5"),
    os.path.join(DATA_DIR, "esca-h5/combined.h5"),
    os.path.join(DATA_DIR, "hnsc-h5/combined.h5"),
    os.path.join(DATA_DIR, "luad-h5/combined.h5"),
    os.path.join(DATA_DIR, "lusc-h5/combined.h5"),
    os.path.join(DATA_DIR, "meso-h5/combined.h5"),
    os.path.join(DATA_DIR, "thym-h5/combined.h5"),
    os.path.join(DATA_DIR, "ucec-h5/combined.h5"),
    os.path.join(DATA_DIR, "uvm-h5/combined.h5"),
]
dataset = TILPatientDataset(data_dir=DATA_DIR, mapping_json=None, h5_paths=H5_PATHS)
train_len = int(0.7 * len(dataset))
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = TILBinaryCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"[DEBUG] Starting epoch {epoch+1} with {len(train_loader)} batches")
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.view(-1)  # Flatten output from [B, 1] to [B]
        labels = labels.float()     # Ensure labels are float for BCELoss
        if outputs.isnan().any():
            print(f"[DEBUG] NaN in model output at epoch {epoch+1}, batch {i}")
        if labels.min() < 0 or labels.max() > 1:
            print(f"[DEBUG] Label out of range at epoch {epoch+1}, batch {i}: min={labels.min().item()}, max={labels.max().item()}")
        try:
            loss = criterion(outputs, labels)
            print(f"[DEBUG] outputs shape: {outputs.shape}, labels shape: {labels.shape}, loss={loss.item():.4f}")
        except RuntimeError as e:
            print(f"[DEBUG] Runtime error at epoch {epoch+1}, batch {i}: {e}")
            print(f"[DEBUG] outputs shape: {outputs.shape}, labels: {labels}")
            raise e
        loss.backward()
        if torch.isnan(loss) or loss.item() == 0:
            print(f"[DEBUG] Warning: Zero or NaN loss detected at epoch {epoch+1}, batch {i}, loss={loss.item()}")
            print(f"[DEBUG] Sample labels: {labels}")
            print(f"[DEBUG] Output logits: {outputs}")
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {total_loss/len(train_loader):.4f}")

# Save trained model
model_save_path = os.path.join(DATA_DIR, "til_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"[INFO] Trained model saved to {model_save_path}")

# Evaluation (with optional GradCAM)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Generate GradCAM visualizations for a few test samples
print("Generating GradCAM visualizations...")
for i, (images, labels) in enumerate(test_loader):
    if i >= 2:  # Only visualize first 2 batches
        break
    images = images.to(DEVICE)
    for j in range(min(4, images.size(0))):  # Visualize up to 4 images per batch
        img_tensor = images[j].unsqueeze(0)
        get_gradcam_visualization(model, img_tensor, layer_name="layer4", save_path=f"gradcam_viz/sample_{i}_{j}.png")

# Optional: Generate GradCAM visualizations on a few test samples
# get_gradcam_visualization(model, sample_image_batch)
