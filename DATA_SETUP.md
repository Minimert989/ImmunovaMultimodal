# Data Setup Instructions

This repository contains the code structure for the Immunova project, but the actual data files have been removed for privacy and size considerations.

## Required Data Files

To run this project, you'll need to add the following data files:

### Main Directory
- `model.pth` - Main model weights
- `patient_data_modality_status.csv` - Patient data and modality status

### Feature Directories
- `label_feature/` - Label dictionary files (*.pkl)
- `omics_feature/` - Omics features (*.pkl)
- `wsi_feature/` - WSI features (*.pkl)

### Module 1 (Immunova2_Module1)
- `til_dataset.h5` - TIL dataset file
- `til_model.pth` - TIL model weights
- Data directories: `acc/`, `blca/`, `brca/`, `cesc/`, `coad/`, `esca/`, `hnsc/`, `luad/`, `lusc/`, `meso/`, `thym/`, `ucec/`, `uvm/`
- H5 directories: `acc-h5/`, `blca-h5/`, `brca-h5/`, etc.

### Module 2 (Immunova2_module2)
- Cancer type directories: `ACC/`, `BLCA/`, `BRCA/`, `CESC/`, `CHOL/`, `COADREAD/`, `DLBC/`, `ESCA/`, `GBM/`, `GBMLGG/`, `HNSC/`, `KICH/`, `KIPAN/`, `KIRC/`, `KIRP/`, `LAML/`, `LGG/`, `LIHC/`, `LUAD/`, `LUSC/`, `MESO/`, `OV/`, `PAAD/`, `PCPG/`, `PRAD/`, `SARC/`, `SKCM/`, `STAD/`, `STES/`, `TGCT/`, `THCA/`, `THYM/`, `UCEC/`, `UCS/`, `UVM/`
- `cptac_zipped/` - CPTAC data files
- Model files: `model_ALL_TCGA_*.pkl`, `model_ALL_TCGA_*.pth`
- Preprocessed files: `labtrans_cuts_ALL_TCGA.npy`, `scaler_ALL_TCGA.joblib`

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r task_requirements.txt`
3. Add your data files to the appropriate directories
4. Update any hardcoded paths in the code to match your data locations
5. Run the desired module:
   - Module 1: `python Immunova2_Module1/train.py`
   - Module 2: `python Immunova2_module2/train.py`
   - Main: `python train.py`

## Notes

- All placeholder files (`*.placeholder`) should be replaced with actual data
- README.md files in data directories indicate where real data should be placed
- The .gitignore file is configured to prevent accidentally committing large data files
