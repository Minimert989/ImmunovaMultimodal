import os
import pandas as pd
import torch
import pickle

# 암종 목록
# 암종 목록 (대문자 그대로 유지)
cancer_types = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COADREAD", "DLBC", "ESCA", "GBM", "GBMLGG",
    "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STAD", "STES", "TGCT",
    "THCA", "THYM", "UCEC", "UCS", "UVM"
]

# Omics 파일 경로 패턴
omics_file_patterns = {
    "rna": "TCGA_rnaseq_{cancer}_immune_markers_with_metadata.csv",
    "methyl": "TCGA_methylation_{cancer}_immune_sites_with_metadata.csv",
    "protein": "TCGA_rppa_{cancer}_immune_proteins_with_metadata.csv",
    "mirna": "TCGA_mirna_{cancer}_immune_markers_with_metadata.csv"
}

# Omics dict 초기화
omics_dict = {
    "rna": {},
    "methyl": {},
    "protein": {},
    "mirna": {}
}

# CSV 루프
base_dir = "Immunova2_module2"
os.makedirs("omics_feature", exist_ok=True)

for cancer in cancer_types:
    cancer_dir = os.path.join(base_dir, cancer.upper())

    for omics_type, pattern in omics_file_patterns.items():
        file_path = os.path.join(cancer_dir, pattern.format(cancer=cancer.upper()))
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, index_col=0)
            df = df.select_dtypes(include=[float, int]).dropna()
            for pid, row in df.iterrows():
                pid_norm = pid
                omics_dict[omics_type][pid_norm] = torch.tensor(row.values, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

# 저장
with open("omics_feature/omics_dict.pkl", "wb") as f:
    pickle.dump(omics_dict, f)

print("✅ omics_dict 저장 완료: omics_feature/omics_dict.pkl")