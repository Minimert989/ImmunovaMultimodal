import os
import pandas as pd
import torch
import pickle

def normalize_pid(pid):
    return pid.replace(".", "-")

cancer_types = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COADREAD", "DLBC", "ESCA", "GBM", "GBMLGG",
    "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STAD", "STES", "TGCT",
    "THCA", "THYM", "UCEC", "UCS", "UVM"
]

label_base_dir = "Immunova2_module2"
os.makedirs("label_feature", exist_ok=True)

for cancer in cancer_types:
    label_dict = {}
    clinical_path = os.path.join(label_base_dir, cancer, f"TCGA_clinical_{cancer}.csv")
    if not os.path.exists(clinical_path):
        print(f"❌ Clinical file missing: {clinical_path}")
        continue

    try:
        df = pd.read_csv(clinical_path, index_col=0)
        df = df.dropna(subset=["status", "overall_survival"])

        for pid, row in df.iterrows():
            pid_norm = normalize_pid(pid)
            try:
                status = 1 if str(row["status"]).strip().lower() == "dead" else 0
                survival = float(row["overall_survival"])
                entry = {
                    "status": torch.tensor([status], dtype=torch.float32),
                    "survival_time": torch.tensor([survival], dtype=torch.float32)
                }

                if "response_label" in df.columns:
                    val = row["response_label"]
                    if pd.notna(val):
                        try:
                            entry["response"] = torch.tensor([int(val)], dtype=torch.float32)
                        except ValueError:
                            pass  # skip if it can't be converted

                label_dict[pid_norm] = entry
            except Exception as e:
                print(f"⚠️ Skipped {pid}: {e}")

        out_path = f"label_feature/label_dict_{cancer}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(label_dict, f)

        print(f"✅ Saved label_dict for {cancer} with {len(label_dict)} entries.")
    except Exception as e:
        print(f"❌ Error loading {clinical_path}: {e}")