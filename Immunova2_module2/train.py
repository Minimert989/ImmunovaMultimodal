import torchtuples as tt
from pycox.models import CoxPH
from pycox.datasets import metabric
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import torchtuples
import torch.nn as nn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
import numpy as np

def is_tcga_dir(dir_name):
    return dir_name.isupper() and not dir_name.startswith("CPTAC") and os.path.isdir(dir_name)

def load_tcga_data(cancer_type):
    cancer_dir = os.path.join(os.getcwd(), cancer_type)
    rnaseq_file = os.path.join(cancer_dir, f"TCGA_rnaseq_{cancer_type}_immune_markers_with_metadata.csv")
    cnv_file = os.path.join(cancer_dir, f"TCGA_copy_number_{cancer_type}_immune_features_with_metadata.csv")
    methylation_file = os.path.join(cancer_dir, f"TCGA_methylation_{cancer_type}_immune_sites_with_metadata.csv")
    mirna_file = os.path.join(cancer_dir, f"TCGA_mirna_{cancer_type}_immune_markers_with_metadata.csv")
    rppa_file = os.path.join(cancer_dir, f"TCGA_rppa_{cancer_type}_immune_proteins_with_metadata.csv")
    clinical_file = os.path.join(cancer_dir, f"TCGA_clinical_{cancer_type}.csv")

    if not os.path.exists(rnaseq_file) or not os.path.exists(clinical_file):
        return None

    try:
        rna_df = pd.read_csv(rnaseq_file)
        key = "case_id" if "case_id" in rna_df.columns else "submitter_id"
        rna_df[key] = rna_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
        print(f"[{cancer_type}] rna_df shape: {rna_df.shape}")
        print(f"[{cancer_type}] rna_df NaNs per column:\n{rna_df.isna().sum()}")
        print(f"[{cancer_type}] rna_df columns:\n{rna_df.columns.tolist()}")

        clinical_df = pd.read_csv(clinical_file)
        clinical_df[key] = clinical_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
        print(f"[{cancer_type}] clinical_df shape: {clinical_df.shape}")
        print(f"[{cancer_type}] clinical_df NaNs per column:\n{clinical_df.isna().sum()}")
        print(f"[{cancer_type}] clinical_df columns:\n{clinical_df.columns.tolist()}")

        if key not in clinical_df.columns:
            return None

        merged = rna_df

        if os.path.exists(cnv_file):
            cnv_df = pd.read_csv(cnv_file)
            cnv_df[key] = cnv_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
            print(f"[{cancer_type}] cnv_df shape: {cnv_df.shape}")
            print(f"[{cancer_type}] cnv_df NaNs per column:\n{cnv_df.isna().sum()}")
            print(f"[{cancer_type}] cnv_df columns:\n{cnv_df.columns.tolist()}")
            merged = pd.merge(merged, cnv_df, on=key, how="left", suffixes=("", "_cnv"))
        if os.path.exists(methylation_file):
            methylation_df = pd.read_csv(methylation_file)
            methylation_df[key] = methylation_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
            print(f"[{cancer_type}] methylation_df shape: {methylation_df.shape}")
            print(f"[{cancer_type}] methylation_df NaNs per column:\n{methylation_df.isna().sum()}")
            print(f"[{cancer_type}] methylation_df columns:\n{methylation_df.columns.tolist()}")
            merged = pd.merge(merged, methylation_df, on=key, how="left", suffixes=("", "_meth"))
        if os.path.exists(mirna_file):
            mirna_df = pd.read_csv(mirna_file)
            mirna_df[key] = mirna_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
            print(f"[{cancer_type}] mirna_df shape: {mirna_df.shape}")
            print(f"[{cancer_type}] mirna_df NaNs per column:\n{mirna_df.isna().sum()}")
            print(f"[{cancer_type}] mirna_df columns:\n{mirna_df.columns.tolist()}")
            merged = pd.merge(merged, mirna_df, on=key, how="left", suffixes=("", "_mirna"))
        if os.path.exists(rppa_file):
            rppa_df = pd.read_csv(rppa_file)
            rppa_df[key] = rppa_df[key].apply(lambda s: ''.join(filter(str.isalnum, str(s))).lower())
            print(f"[{cancer_type}] rppa_df shape: {rppa_df.shape}")
            print(f"[{cancer_type}] rppa_df NaNs per column:\n{rppa_df.isna().sum()}")
            print(f"[{cancer_type}] rppa_df columns:\n{rppa_df.columns.tolist()}")
            merged = pd.merge(merged, rppa_df, on=key, how="left", suffixes=("", "_rppa"))

        merged = pd.merge(merged, clinical_df, on=key)

        # Standardize clinical columns if present
        clinical_cols = ['status', 'overall_survival', 'years_to_birth', 'gender', 'pathologic_stage', 'pathology_t_stage', 'pathology_n_stage', 'radiation_therapy', 'residual_tumor']
        for col in clinical_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype(str).str.strip()

        # Map status, with debug print and dtype/unique checks
        if 'status' in merged.columns:
            print(f"[{cancer_type}] status dtype:", merged['status'].dtype)
            print(f"[{cancer_type}] status unique values BEFORE mapping:", merged['status'].unique())
        if 'status' in merged.columns:
            if merged['status'].dtype == object or merged['status'].dtype.name == 'category':
                merged['status'] = merged['status'].astype(str).str.strip()
                merged['status'] = merged['status'].map({'Dead': 1, 'Alive': 0})

        # Convert overall_survival to numeric, clean commas and non-numeric chars
        if 'overall_survival' in merged.columns:
            merged['overall_survival'] = merged['overall_survival'].str.replace(",", ".", regex=False)
            merged['overall_survival'] = pd.to_numeric(merged['overall_survival'], errors='coerce')

        # Derive age from years_to_birth
        if 'years_to_birth' in merged.columns:
            merged['years_to_birth'] = pd.to_numeric(merged['years_to_birth'], errors='coerce')
            merged['age'] = 2024 - merged['years_to_birth']

        # Map gender
        if 'gender' in merged.columns:
            merged['gender'] = merged['gender'].str.upper().map({'MALE': 0, 'FEMALE': 1})

        # Map pathologic_stage categorical to numeric
        if 'pathologic_stage' in merged.columns:
            stage_map = {'stage i':1, 'stage ia':1, 'stage ib':1, 'stage ii':2, 'stage iia':2, 'stage iib':2, 'stage iii':3, 'stage iiia':3, 'stage iiib':3, 'stage iv':4}
            merged['pathologic_stage'] = merged['pathologic_stage'].str.lower().map(stage_map)

        # Map pathology_t_stage categorical to numeric
        if 'pathology_t_stage' in merged.columns:
            t_stage_map = {'t0':0, 't1':1, 't1a':1, 't1b':1, 't2':2, 't2a':2, 't2b':2, 't3':3, 't3a':3, 't3b':3, 't4':4}
            merged['pathology_t_stage'] = merged['pathology_t_stage'].str.lower().map(t_stage_map)

        # Map pathology_n_stage categorical to numeric
        if 'pathology_n_stage' in merged.columns:
            n_stage_map = {'n0':0, 'n1':1, 'n1a':1, 'n1b':1, 'n2':2, 'n2a':2, 'n2b':2, 'n3':3}
            merged['pathology_n_stage'] = merged['pathology_n_stage'].str.lower().map(n_stage_map)

        # Map radiation_therapy categorical to numeric
        if 'radiation_therapy' in merged.columns:
            radiation_map = {'yes':1, 'no':0}
            merged['radiation_therapy'] = merged['radiation_therapy'].str.lower().map(radiation_map)

        # Map residual_tumor categorical to numeric
        if 'residual_tumor' in merged.columns:
            residual_map = {'r0':0, 'r1':1, 'r2':2, 'rx':-1}
            merged['residual_tumor'] = merged['residual_tumor'].str.lower().map(residual_map)

        # Fill NA with -1 in clinical columns
        for col in clinical_cols + ['age']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(-1)

        # Drop columns with over 50% NaN if present
        thresh = len(merged) * 0.5
        merged = merged.loc[:, merged.isnull().sum() <= thresh]

        if "overall_survival" in merged.columns and "status" in merged.columns:
            merged = merged.dropna(subset=["overall_survival", "status"])
            return merged, key, ("overall_survival", "status")
        else:
            return None
    except Exception as e:
        print(f"[{cancer_type}] 오류 발생: {e}")
        return None

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def train_tcga_model(merged, key, cancer_type, label_col):
    exclude_cols = [key]
    if isinstance(label_col, tuple):
        exclude_cols.extend(label_col)
    else:
        exclude_cols.append(label_col)
    exclude_cols.append("overall_survival")
    feature_cols = [col for col in merged.columns if col not in exclude_cols and merged[col].dtype != "object"]
    X = merged[feature_cols]
    if isinstance(label_col, tuple):
        y = merged[label_col[0]]  # For regression, use duration
    else:
        y = merged[label_col]

    # Check if XGBoost model already exists before training
    model_path = f"model_{cancer_type}_XGBoost.pkl"
    if os.path.exists(model_path):
        print(f"✅ {cancer_type} XGBoost 모델이 이미 존재합니다. 재학습을 건너뜁니다.")
        return

    print(f"[{cancer_type}] 초기 X shape: {X.shape}, y shape: {y.shape}")

    X = X.fillna(0)
    y = y.loc[X.index]

    print(f"[{cancer_type}] NaN 제거 후 X shape: {X.shape}, y shape: {y.shape}")

    print(f"[{cancer_type}] NaN 포함 열 개수: {(X.isna().sum() > 0).sum()} / 전체 열: {X.shape[1]}")

    print(f"[{cancer_type}] y 라벨 값 분포:\n{y.value_counts(dropna=False)}")
    print(f"[{cancer_type}] X 샘플 수: {X.shape[0]}, Feature 수: {X.shape[1]}")
    print(f"[{cancer_type}] y 고유값: {y.unique()}")
    print(f"[{cancer_type}] y 값 중 NaN 개수: {y.isna().sum()}, inf 개수: {np.isinf(y).sum()}, -inf 개수: {np.isneginf(y).sum()}, 너무 큰 값 개수: {(y > 1e10).sum()}")

    y = y.replace(-1.0, 1.0)
    y = y.apply(lambda x: np.log1p(x))
    print(f"[{cancer_type}] y after log1p - NaNs: {y.isna().sum()}, Infs: {np.isinf(y).sum()}")

    valid_idx = y.notna() & np.isfinite(y)
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"[{cancer_type}] log1p 이후 유효 샘플 수: {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    models = {
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist"),
    }

    for name, model in models.items():
        # Note: scikit-learn models do not use GPU. To accelerate, switch to PyTorch models if needed.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n📈 [{cancer_type}] {name} MSE: {mse:.4f}, R^2: {r2:.4f}")

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = importances.argsort()[::-1][:20]
            plt.figure(figsize=(8,5))
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
            plt.title(f"{cancer_type} {name} - Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(f"feature_importance_{cancer_type}_{name}.png")
            plt.close()

        # Reduce features via top 500 importances
        importances = model.feature_importances_
        top_indices = importances.argsort()[::-1][:500]
        X = X.iloc[:, top_indices]

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"🔁 {name} 5-fold CV 평균 R^2: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Save model and predictions
        joblib.dump(model, model_path)
        pred_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
        pred_df.to_csv(f"predictions_{cancer_type}_{name}.csv", index=False)

        print(f"🎯 Finished training {name} on device: {device}")


# DeepSurv 모델 훈련 함수 추가
def train_deepsurv_model(X, durations, events, cancer_type):
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pycox.models import LogisticHazard

    # Duration and event setup
    durations = durations.apply(lambda x: np.log1p(x))
    print(f"[{cancer_type}] durations after log1p - NaNs: {durations.isna().sum()}, Infs: {np.isinf(durations).sum()}")
    durations = durations.values.astype(np.float32)
    events = events.values.astype(int)

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.astype(np.float32)
    X_train, X_test, y_train_duration, y_test_duration, y_train_event, y_test_event = train_test_split(
        X_scaled, durations, events, test_size=0.2, random_state=42
    )

    labtrans = LogisticHazard.label_transform(10)
    from joblib import dump
    dump(scaler, f"scaler_{cancer_type}.joblib")

    y_train = labtrans.fit_transform(y_train_duration, y_train_event)
    y_test = labtrans.transform(y_test_duration, y_test_event)
    # Save labtrans.cuts and scaler (after fitting and transforming)
    if labtrans.cuts is not None:
        np.save(f"labtrans_cuts_{cancer_type}.npy", labtrans.cuts.tolist(), allow_pickle=True)
    else:
        print(f"❗ Warning: labtrans.cuts is None for {cancer_type}, skipping save.")

    train_data = (X_train, y_train)
    val_data = (X_test, y_test)

    net = tt.practical.MLPVanilla(in_features=X.shape[1], num_nodes=[32, 32], out_features=labtrans.out_features, dropout=0.1, activation=nn.ReLU)
    model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.fit(*train_data, batch_size=256, epochs=512, val_data=val_data, verbose=True)

    # Evaluation
    surv = model.predict_surv_df(X_test)
    surv.to_csv(f"survival_predictions_{cancer_type}_deepsurv.csv")
    # Save only the model's state_dict to avoid pickle error
    torch.save(model.net.state_dict(), f"model_{cancer_type}_deepsurv.pth")

    print(f"✅ DeepSurv 모델 훈련 완료 및 저장됨 (암종: {cancer_type})")

if __name__ == "__main__":
    retrain_xgboost = True  # True to retrain XGBoost
    retrain_deepsurv = True  # Always retrain DeepSurv
    dirs = [d for d in os.listdir() if is_tcga_dir(d)]
    print(f"TCGA 암종 폴더 감지됨: {dirs}")

    combined_df = []
    used_key = None
    used_label_col = None

    for cancer in dirs:
        result = load_tcga_data(cancer)
        if result:
            merged, key, label_col = result
            combined_df.append(merged)
            used_key = key
            used_label_col = label_col

    if combined_df:
        full_data = pd.concat(combined_df, ignore_index=True)
        # Filter out samples with overall_survival == -1.0
        full_data = full_data[full_data["overall_survival"] != -1.0]
        if isinstance(used_label_col, tuple):
            print("전체 통합 라벨 분포:")
            for col in used_label_col:
                print(f"{col} 분포:\n{full_data[col].value_counts(dropna=False)}")
        else:
            print("전체 통합 responder_label 라벨 분포:\n", full_data[used_label_col].value_counts(dropna=False))
        print("전체 데이터 shape:", full_data.shape)
        print("사용된 키:", used_key)
        if isinstance(used_label_col, tuple):
            with open("used_label_col.txt", "w") as f:
                f.write(','.join(used_label_col))
        else:
            with open("used_label_col.txt", "w") as f:
                f.write(used_label_col)

        clinical_features = ['years_to_birth', 'gender', 'pathologic_stage', 'pathology_t_stage', 'pathology_n_stage', 'radiation_therapy', 'residual_tumor']
        for col in clinical_features:
            if col in full_data.columns:
                if full_data[col].dtype == "object":
                    full_data[col] = full_data[col].astype("category").cat.codes
                full_data[col] = full_data[col].fillna(-1)

        # XGBoost 모델 경로
        model_path = f"model_ALL_TCGA_XGBoost.pkl"
        if retrain_xgboost or not os.path.exists(model_path):
            train_tcga_model(full_data, used_key, "ALL_TCGA", used_label_col)
        else:
            print("✅ XGBoost 모델이 이미 존재합니다. 재학습을 건너뜁니다.")

        # DeepSurv 모델 훈련 추가
        import os
        feature_cols = [col for col in full_data.columns if col not in [used_key, *used_label_col, "overall_survival"] and full_data[col].dtype != "object"]
        duration_col, event_col = used_label_col
        deepsurv_model_path = f"model_ALL_TCGA_deepsurv.pth"
        if retrain_deepsurv or not os.path.exists(deepsurv_model_path):
            train_deepsurv_model(full_data[feature_cols], full_data[duration_col], full_data[event_col], "ALL_TCGA")
        else:
            print("✅ DeepSurv 모델이 이미 존재합니다. 재학습을 건너뜁니다.")

        # DeepSurv 모델 평가
        from pycox.evaluation import EvalSurv
        from pycox.models import LogisticHazard
        import pandas as pd
        import torch
        import torchtuples as tt
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from joblib import load

        # Prepare evaluation data
        scaler = load(f"scaler_ALL_TCGA.joblib")
        X_scaled = scaler.transform(full_data[feature_cols].astype(np.float32))
        durations = np.log1p(full_data[duration_col].values.astype(np.float32))
        events = full_data[event_col].values.astype(int)

        X_train, X_test, y_train_duration, y_test_duration, y_train_event, y_test_event = train_test_split(
            X_scaled, durations, events, test_size=0.2, random_state=42
        )

        # Load cuts for correct duration_index
        cuts = np.load(f"labtrans_cuts_ALL_TCGA.npy", allow_pickle=True).tolist()
        if cuts is None:
            raise ValueError("Loaded cuts is None. Check if the file is valid.")
        # Re-create labtrans and cuts for correct duration_index
        labtrans = LogisticHazard.label_transform(len(cuts) if cuts is not None else 10)
        labtrans.cuts = cuts

        # Load model with corrected duration index
        net = tt.practical.MLPVanilla(in_features=len(feature_cols), num_nodes=[32, 32], out_features=10, dropout=0.1, activation=nn.ReLU)
        model = LogisticHazard(net, tt.optim.Adam, duration_index=cuts)
        model.net.load_state_dict(torch.load(f"model_ALL_TCGA_deepsurv.pth"))

        # Evaluate
        surv = model.predict_surv_df(X_test)
        import matplotlib.pyplot as plt
        surv.iloc[:, :10].plot()
        plt.title("Sample Survival Curves")
        plt.xlabel("Time Index")
        plt.ylabel("Survival Probability")
        plt.tight_layout()
        plt.savefig("sample_survival_curves.png")
        ev = EvalSurv(surv, y_test_duration, y_test_event, censor_surv='km')
        c_index = ev.concordance_td('antolini')
        ibs = ev.integrated_brier_score(np.linspace(0, durations.max(), 100))
        print(f"\n📊 DeepSurv 평가 결과 - C-index: {c_index:.4f}, IBS: {ibs:.4f}")
    else:
        print("❗ 사용할 수 있는 데이터가 없습니다.")

# ==========================
# Random Survival Forest (RSF) with sksurv
# ==========================
if 'full_data' in locals() and full_data is not None and 'feature_cols' in locals():
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sksurv.metrics import concordance_index_censored
        # Prepare structured array for sksurv
        y_structured = Surv.from_arrays(event=full_data[event_col].astype(bool), time=full_data[duration_col])
        X_rsf = full_data[feature_cols].astype(np.float32)
        X_rsf = X_rsf.fillna(0)
        scaler_rsf = StandardScaler()
        X_rsf_scaled = scaler_rsf.fit_transform(X_rsf)
        X_train_rsf, X_test_rsf, y_train_rsf, y_test_rsf = train_test_split(X_rsf_scaled, y_structured, test_size=0.2, random_state=42)
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15,
                                   max_features="sqrt", n_jobs=-1, random_state=42)
        rsf.fit(X_train_rsf, y_train_rsf)
        pred_surv = rsf.predict(X_test_rsf)
        c_index_rsf = concordance_index_censored(y_test_rsf['event'], y_test_rsf['time'], pred_surv)[0]
        print(f"🌲 RSF 평가 결과 - C-index: {c_index_rsf:.4f}")
    except Exception as e:
        print(f"❗ RSF 평가 중 오류 발생: {e}")