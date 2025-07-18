import torch
import os
import logging
import time
import pickle
import glob
from tqdm import tqdm

import torchvision.models as models

# --- 1. LOGGING SETUP ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('wsi_feature_extraction.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


# --- 2. MODEL AND DEVICE SETUP ---
logger.info("Setting up the model and device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # feature vector only
model.eval().to(device)
logger.info("ResNet-18 model loaded in evaluation mode.")

# --- 3. SCRIPT CONFIGURATION ---
cancer_types = ["acc", "blca", "brca", "cesc", "coad", "esca", "hnsc", "luad", "lusc", "meso", "thym", "ucec", "uvm"]
base_dir = "Immunova2_Module1"
output_dir = "wsi_feature"

os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory '{output_dir}' is ready.")

# --- 4. FEATURE EXTRACTION LOOP ---
script_start_time = time.time()
logger.info("--- Starting WSI Feature Extraction ---")

for cancer in cancer_types:
    cancer_start_time = time.time()
    logger.info(f"Processing cancer type: {cancer.upper()}")
    
    pt_files = glob.glob(os.path.join(base_dir, cancer, "*.pt"))
    if not pt_files:
        logger.warning(f"No .pt files found for {cancer}. Skipping.")
        continue
        
    logger.info(f"Found {len(pt_files)} patient files for {cancer}.")
    wsi_features = {}

    with torch.no_grad():
        for pt_path in tqdm(pt_files, desc=f"Extracting {cancer.upper()}", unit="file"):
            patient_start_time = time.time()
            patient_id = os.path.basename(pt_path).replace(".pt", "")
            
            try:
                data = torch.load(pt_path, map_location=device)
                patches = data["images"]  # Expected shape: (N, 3, 224, 224)
                
                if patches.ndim != 4 or patches.shape[1] != 3:
                    logger.error(f"Skipping {patient_id}: Invalid patch dimensions {patches.shape}")
                    continue

                feats = model(patches)
                slide_feat = feats.cpu()
                wsi_features[patient_id] = slide_feat
                
                patient_end_time = time.time()
                logger.debug(f"Successfully processed {patient_id} with {len(patches)} patches in {patient_end_time - patient_start_time:.2f} seconds.")

            except Exception as e:
                logger.error(f"Failed to process {pt_path} for patient {patient_id}. Reason: {e}", exc_info=True)

    # --- 5. SAVE FEATURES ---
    if not wsi_features:
        logger.warning(f"No features were extracted for {cancer}. No .pkl file will be saved.")
        continue

    out_path = os.path.join(output_dir, f"wsi_features_{cancer.upper()}.pkl")
    try:
        with open(out_path, "wb") as f:
            pickle.dump(wsi_features, f)
        cancer_end_time = time.time()
        logger.info(f"✅ Saved WSI features for {cancer.upper()} to {out_path}. Took {cancer_end_time - cancer_start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to save .pkl file for {cancer.upper()} at {out_path}. Reason: {e}", exc_info=True)

script_end_time = time.time()
logger.info(f"--- Finished WSI Feature Extraction in {script_end_time - script_start_time:.2f} seconds ---")
