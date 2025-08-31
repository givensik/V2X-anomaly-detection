# test_autoencoder.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, precision_score, recall_score, f1_score
)

from z_models.lstm_autoencoder import LSTMAutoEncoder


def extract_sequence_scenario_ids(df: pd.DataFrame, window_size: int = 30) -> np.ndarray:
    scenario_ids = []
    for _, group in df.groupby(["station_id", "scenario_id"]):
        group = group.sort_values("timestamp")
        current_ids = group["scenario_id"].values
        for i in range(len(current_ids) - window_size + 1):
            scenario_ids.append(current_ids[i + window_size // 2])
    return np.array(scenario_ids)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 📁 ensure output folder
os.makedirs("z_results", exist_ok=True)

# 📦 Load Model
model = LSTMAutoEncoder(input_dim=6).to(DEVICE)
model.load_state_dict(torch.load("z_models/autoencoder.pth"))
model.eval()

# 📊 Load Test Data
X_test = np.load("z_data/test_X.npy")
y_test = np.load("z_data/test_y.npy")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# 📂 Load raw CSV for scenario alignment
df_veremi = pd.read_csv("z_test/veremi_etsi_standardized.csv")
scenario_ids = extract_sequence_scenario_ids(df_veremi, window_size=30)

# 🔎 Inference
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()

# 🎯 Set Threshold
threshold = np.percentile(errors[y_test == 0], 90)
y_pred = (errors > threshold).astype(int)

# 📈 Metrics
roc_auc = roc_auc_score(y_test, errors)
precision, recall, _ = precision_recall_curve(y_test, errors)
pr_auc = auc(recall, precision)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
print(f"Threshold: {threshold:.4f}")

# 📊 Visualization
plt.figure(figsize=(8, 4))
sns.histplot(errors[y_test == 0], label="Normal", color="blue", bins=100, stat="density")
sns.histplot(errors[y_test == 1], label="Attack", color="red", bins=100, stat="density")
plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("z_results/reconstruction_error_distribution.png")
plt.show()

# 🧪 Scenario-wise Evaluation
scenario_metrics = []

for scenario in np.unique(scenario_ids):
    idx = scenario_ids == scenario
    y_true = y_test[idx]
    y_score = errors[idx]
    y_pred = (y_score > threshold).astype(int)  # ← 이 줄이 핵심 수정

    if len(np.unique(y_true)) < 2:
        continue  # 한 클래스만 존재하면 스킵

    roc = roc_auc_score(y_true, y_score)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_vals, precision_vals)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    scenario_metrics.append({
        "scenario_id": scenario,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "support_attack": int(sum(y_true)),
        "support_normal": int(len(y_true) - sum(y_true))
    })



metrics_df = pd.DataFrame(scenario_metrics)
print(metrics_df.sort_values("f1_score", ascending=False))
metrics_df.to_csv("z_results/scenario_evaluation.csv", index=False)

# 시나리오별 F1-score 바 차트
plt.figure(figsize=(12, 6))
metrics_df_sorted = metrics_df.sort_values("f1_score", ascending=False)

plt.bar(metrics_df_sorted["scenario_id"], metrics_df_sorted["f1_score"], color='skyblue')
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("F1-score")
plt.title("Scenario-wise F1-score (sorted)")
plt.tight_layout()
plt.savefig("z_results/scenario_f1_score.png")
plt.show()

plt.figure(figsize=(14, 6))
x = np.arange(len(metrics_df_sorted))

plt.plot(x, metrics_df_sorted["precision"], marker='o', label="Precision")
plt.plot(x, metrics_df_sorted["recall"], marker='s', label="Recall")
plt.plot(x, metrics_df_sorted["f1_score"], marker='^', label="F1-score")

plt.xticks(x, metrics_df_sorted["scenario_id"], rotation=90, fontsize=8)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Scenario-wise Precision / Recall / F1-score")
plt.legend()
plt.tight_layout()
plt.savefig("z_results/scenario_all_metrics.png")
plt.show()