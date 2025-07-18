

import pandas as pd
import numpy as np
import torch
import torchtuples as tt
import matplotlib.pyplot as plt

from pycox.models import DeepSurv
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split

# ✅ Load processed data (replace this with your own loading logic)
full_data = pd.read_csv("ALL_TCGA.csv")  # or your final processed dataset
used_label_col = ("overall_survival", "status")  # adjust if needed

X = full_data.drop(columns=["case_id", "overall_survival", "status"])
durations = full_data["overall_survival"]
events = full_data["status"]

# Convert -1.0 to NaN and drop (as done in train.py)
valid_mask = (durations != -1.0)
X = X.loc[valid_mask]
durations = durations.loc[valid_mask]
events = events.loc[valid_mask]

# Split data
X_train, X_test, durations_train, durations_test, events_train, events_test = train_test_split(
    X, durations, events, test_size=0.2, random_state=42
)

# Log-transform durations
durations_train = np.log1p(durations_train)
durations_test = np.log1p(durations_test)

# Load trained DeepSurv model
model = DeepSurv.load_net("trained_deepsurv_model.pt")

# Predict survival
surv = model.predict_surv_df(X_test)

# Evaluate
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
c_index = ev.concordance_td('antolini')
ibs = ev.integrated_brier_score(np.linspace(durations_test.min(), durations_test.max(), 100))

print(f"✅ DeepSurv Evaluation Results:")
print(f"   - Concordance Index (C-index): {c_index:.4f}")
print(f"   - Integrated Brier Score (IBS): {ibs:.4f}")

# Optional plot
surv.iloc[:, :10].plot()
plt.title("Sample Survival Curves")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.show()