"""
Configuration file for Immunova pipeline
"""

# Model Configuration
MODEL_CONFIG = {
    "hidden_dim": 512,
    "n_heads": 4,
    "n_layers": 2,
    "til_classes": 4,
    "dropout": 0.3
}

# Input Dimensions
INPUT_DIMS = {
    "rna": 1000,
    "methyl": 512,
    "protein": 256,
    "mirna": 128,
    "wsi": 512
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10,  # Early stopping patience
    "val_split": 0.2,
    "random_seed": 42
}

# Task Configuration
TASKS = {
    "til": {
        "type": "classification",
        "num_classes": 4,
        "loss_fn": "bce_with_logits"
    },
    "response": {
        "type": "classification",
        "num_classes": 1,
        "loss_fn": "bce_with_logits"
    },
    "survival": {
        "type": "regression",
        "num_classes": 1,
        "loss_fn": "mse"
    }
}

# Data Paths
DATA_PATHS = {
    "wsi_feature": "wsi_feature/",
    "omics_feature": "omics_feature/",
    "label_feature": "label_feature/",
    "module1": "Immunova2_Module1/",
    "module2": "Immunova2_module2/"
}

# Output Paths
OUTPUT_PATHS = {
    "model": "model.pth",
    "predictions": "predictions_{task}.csv",
    "val_ids": "val_ids.txt",
    "training_log": "training.log",
    "visualization": "plots/"
}

# Supported Cancer Types
CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "COADREAD", "DLBC", 
    "ESCA", "GBM", "GBMLGG", "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", 
    "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", 
    "PCPG", "PRAD", "SARC", "SKCM", "STAD", "STES", "TGCT", "THCA", 
    "THYM", "UCEC", "UCS", "UVM"
]

# Visualization Settings
VIZ_CONFIG = {
    "figsize": (10, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": {
            "filename": "immunova.log",
            "mode": "a"
        },
        "console": {
            "level": "INFO"
        }
    }
}
