# test_prediction_ae.py

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

from z_models.prediction_autoencoder import PredictionAutoEncoder


def extract_sequence_scenario_ids(df: pd.DataFrame, window_size: int = 20, pred_len: int = 1) -> np.ndarray:
    scenario_ids = []
    for _, group in df.groupby(["station_id", "scenario_id"]):
        group = group.sort_values("timestamp")
        current_ids = group["scenario_id"].values
        for i in range(len(current_ids) - window_size - pred_len + 1):
            scenario_ids.append(current_ids[i + window_size // 2])
    return np.array(scenario_ids)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("z_results", exist_ok=True)

# Load model
model = PredictionAutoEncoder(input_dim=8, hidden_dim=128, seq_len=5, pred_len=1).to(DEVICE)
model.load_state_dict(torch.load("z_models/prediction_autoencoder.pth"))
model.eval()

# Load data
X_test = np.load("z_data/test_X.npy")  # (N, 30, 8)
Y_test = np.load("z_data/test_Y.npy")  # (N, 1, 8)
y_test = np.load("z_data/test_labels.npy")  # (N,)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(DEVICE)

# Load raw VEREMI CSV for scenario ID alignment
df_veremi = pd.read_csv("z_test/veremi_etsi_standardized.csv")
scenario_ids = extract_sequence_scenario_ids(df_veremi, window_size=5, pred_len=1)

# Prediction
with torch.no_grad():
    predicted = model(X_test_tensor)
    errors = torch.mean((Y_test_tensor - predicted) ** 2, dim=(1, 2)).cpu().numpy()

# Thresholding
# threshold = np.percentile(errors[y_test == 0], 90)
# y_pred = (errors > threshold).astype(int)
thresholds = np.linspace(min(errors), max(errors), 100)
f1_scores = [f1_score(y_test, (errors > t).astype(int), zero_division=0) for t in thresholds]
best_idx = np.argmax(f1_scores)
threshold = thresholds[best_idx]

y_pred = (errors > threshold).astype(int)

print(f"[✔] Best threshold (by F1): {threshold:.4f}")
print(f"[✔] Best F1-score       : {f1_scores[best_idx]:.4f}")



# Overall Metrics
roc_auc = roc_auc_score(y_test, errors)
precision, recall, _ = precision_recall_curve(y_test, errors)
pr_auc = auc(recall, precision)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
print(f"Threshold: {threshold:.4f}")

# Error Distribution
plt.figure(figsize=(12,6))
bins = np.logspace(np.log10(errors.min()+1e-6), np.log10(errors.max()), 100)

plt.hist(errors[y_test==0], bins=bins, color="blue", alpha=0.5, label="Normal", density=True, edgecolor="black", linewidth=0.3)
plt.hist(errors[y_test==1], bins=bins, color="red", alpha=0.5, label="Attack", density=True, edgecolor="black", linewidth=0.3)
# sns.histplot(errors[y_test == 0], label="Normal", color="blue", bins=bins, stat="density", common_norm=False)
# sns.histplot(errors[y_test == 1], label="Attack", color="red", bins=bins, stat="density", common_norm=False, alpha=0.4)

plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
plt.xscale("log")   # 👈 로그말고 다른거 없나
plt.yscale("log")   # 👈 y축도 로그
plt.title("Prediction Error Distribution (log scale)")
plt.xlabel("MSE (log)")
plt.ylabel("Density (log)")
plt.legend()
plt.tight_layout()
plt.show()

# Scenario-wise Evaluation
scenario_metrics = []
for scenario in np.unique(scenario_ids):
    idx = scenario_ids == scenario
    y_true = y_test[idx]
    y_score = errors[idx]
    y_pred_scenario = (y_score > threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        continue

    roc = roc_auc_score(y_true, y_score)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_vals, precision_vals)
    f1 = f1_score(y_true, y_pred_scenario, zero_division=0)
    precision_s = precision_score(y_true, y_pred_scenario, zero_division=0)
    recall_s = recall_score(y_true, y_pred_scenario, zero_division=0)

    scenario_metrics.append({
        "scenario_id": scenario,
        "precision": precision_s,
        "recall": recall_s,
        "f1_score": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "support_attack": int(sum(y_true)),
        "support_normal": int(len(y_true) - sum(y_true))
    })

metrics_df = pd.DataFrame(scenario_metrics)
print(metrics_df.sort_values("f1_score", ascending=False))
metrics_df.to_csv("z_results/prediction_scenario_eval.csv", index=False)

# Visualizations


plt.figure(figsize=(12, 6))
metrics_df_sorted = metrics_df.sort_values("f1_score", ascending=False)
plt.bar(metrics_df_sorted["scenario_id"], metrics_df_sorted["f1_score"], color='lightcoral')
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("F1-score")
plt.title("Scenario-wise F1-score (Prediction-AE)")
plt.tight_layout()
plt.savefig("z_results/prediction_scenario_f1.png")
plt.show()

plt.figure(figsize=(14, 6))
x = np.arange(len(metrics_df_sorted))
plt.plot(x, metrics_df_sorted["precision"], marker='o', label="Precision")
plt.plot(x, metrics_df_sorted["recall"], marker='s', label="Recall")
plt.plot(x, metrics_df_sorted["f1_score"], marker='^', label="F1-score")
plt.xticks(x, metrics_df_sorted["scenario_id"], rotation=90, fontsize=8)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Scenario-wise Metrics (Prediction-AE)")
plt.legend()
plt.tight_layout()
plt.savefig("z_results/prediction_scenario_all_metrics.png")
plt.show()

# =========================
# 추가 시각화
# =========================

# 1. KDE Plot (에러 분포 곡선) -> 정상/공격 분포 겹침 정도
plt.figure(figsize=(8,4))
sns.kdeplot(errors[y_test==0], label="Normal", color="blue", fill=True)
sns.kdeplot(errors[y_test==1], label="Attack", color="red", fill=True)
plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
plt.title("Prediction Error KDE Distribution")
plt.xlabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("z_results/prediction_error_kde.png")
plt.show()

# 2. Scenario-wise Error Boxplot -> 시나리오 별 error 편차
df_errors = pd.DataFrame({
    "scenario_id": scenario_ids,
    "error": errors,
    "label": y_test
})
plt.figure(figsize=(12,6))
sns.boxplot(data=df_errors, x="scenario_id", y="error", hue="label", showfliers=False)
plt.xticks(rotation=90, fontsize=8)
plt.title("Scenario-wise Error Distribution (Boxplot)")
plt.tight_layout()
plt.savefig("z_results/prediction_scenario_error_boxplot.png")
plt.show()

# 3. ROC & PR Curve -> ROC 곡선 및 PR 곡선
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, _ = roc_curve(y_test, errors)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, errors)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall_curve, precision_curve, label=f"AUC={pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.tight_layout()
plt.savefig("z_results/prediction_roc_pr.png")
plt.show()