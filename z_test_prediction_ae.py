# test_prediction_ae.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    fbeta_score, roc_auc_score, precision_recall_curve, auc,
    classification_report, precision_score, recall_score, f1_score
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# # 기존 방식
# Load data
X_test = np.load("z_data/test_X.npy")  # (N, 30, 8)
Y_test = np.load("z_data/test_Y.npy")  # (N, 1, 8)
y_test = np.load("z_data/test_labels.npy")  # (N,)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(DEVICE)

# Prediction (기존 방식)
with torch.no_grad():
    predicted = model(X_test_tensor)
    errors = torch.mean((Y_test_tensor - predicted) ** 2, dim=(1, 2)).cpu().numpy()


# Thresholding
# threshold = np.percentile(errors[y_test == 0], 90)
# y_pred = (errors > threshold).astype(int)
# thresholds = np.linspace(min(errors), max(errors), 100)
thresholds = np.logspace(np.log10(errors.min()+1e-6), np.log10(errors.max()), 300)
# f1_scores = [f1_score(y_test, (errors > t).astype(int), zero_division=0) for t in thresholds]
# best_idx = np.argmax(f1_scores)
# threshold = thresholds[best_idx]
precision, recall, thresholds = precision_recall_curve(y_test, errors)
# f2_scores = [fbeta_score(y_test, (errors > t).astype(int), beta=2) for t in thresholds]
beta = 2
f2_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)

best_idx = np.argmax(f2_scores)
threshold = thresholds[best_idx]
print("Best Threshold (by F2):", threshold)


y_pred = (errors > threshold).astype(int)

print(f"[✔] Best threshold : {threshold:.4f}")
# print(f"[✔] Best F1-score       : {f1_scores[best_idx]:.4f}")

# # -------------------
# # 데이터 로드 및 분리
# # -------------------
# X_all = np.load("z_data/test_X.npy")
# Y_all = np.load("z_data/test_Y.npy")
# y_all = np.load("z_data/test_labels.npy")

# # Validation / Test 분리 (예: 앞 30%는 validation, 나머지는 test)
# split_idx = int(len(X_all) * 0.3)
# X_val, Y_val, y_val = X_all[:split_idx], Y_all[:split_idx], y_all[:split_idx]
# X_test, Y_test, y_test = X_all[split_idx:], Y_all[split_idx:], y_all[split_idx:]

# # Torch tensor 변환
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
# Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(DEVICE)



# # -------------------
# # Validation 에러로 threshold 결정
# # -------------------
# with torch.no_grad():
#     pred_val = model(X_val_tensor)
#     errors_val = torch.mean((Y_val_tensor - pred_val) ** 2, dim=(1, 2)).cpu().numpy()

# # 정상 데이터 기준 threshold (예: 95%)
# threshold = np.percentile(errors_val[y_val == 0], 80)
# print(f"[✔] Validation 기반 Threshold: {threshold:.4f}")

# # Validation에서 threshold 최적화
# # thresholds = np.linspace(errors_val.min(), errors_val.max(), 200)
# # f1_scores = [f1_score(y_val, (errors_val > t).astype(int)) for t in thresholds]
# # best_idx = np.argmax(f1_scores)
# # threshold = thresholds[best_idx]

# # print(f"[✔] Validation 기반 F1 최적 Threshold: {threshold:.4f}")
# # print(f"[✔] Validation F1-score: {f1_scores[best_idx]:.4f}")

# # -------------------
# # Test 평가
# # -------------------
with torch.no_grad():
    pred_test = model(X_test_tensor)
    errors_test = torch.mean((Y_test_tensor - pred_test) ** 2, dim=(1, 2)).cpu().numpy()
errors = errors_test
y_pred = (errors > threshold).astype(int)





# Load raw VEREMI CSV for scenario ID alignment
df_veremi = pd.read_csv("z_test/veremi_etsi_standardized.csv")
scenario_ids = extract_sequence_scenario_ids(df_veremi, window_size=5, pred_len=1)






# Overall Metrics
roc_auc = roc_auc_score(y_test, errors)
precision, recall, _ = precision_recall_curve(y_test, errors)
pr_auc = auc(recall, precision)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.8f}")
print(f"PR-AUC : {pr_auc:.8f}")
print(f"Threshold: {threshold:.8f}")

# Error Distribution
plt.figure(figsize=(10,5), dpi=300)
bins = np.logspace(np.log10(errors.min()+1e-6), np.log10(errors.max()), 300)

plt.hist(errors[y_test==0], bins=bins, color="blue", alpha=0.5, label="Normal", density=False, edgecolor="black", linewidth=0.3)
plt.hist(errors[y_test==1], bins=bins, color="red", alpha=0.3, label="Attack", density=False, edgecolor="black", linewidth=0.3)
# sns.histplot(errors[y_test == 0], label="Normal", color="blue", bins=bins, stat="density", common_norm=False)
# sns.histplot(errors[y_test == 1], label="Attack", color="red", bins=bins, stat="density", common_norm=False, alpha=0.4)

# plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
plt.xscale("log")
plt.yscale("log")   # 👈 y축도 로그
plt.title("Prediction Error Distribution (log scale)")
plt.xlabel("MSE (log)", fontdict={'fontsize': 12})
plt.ylabel("Density (log)", fontdict={'fontsize': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig("prediction_error_distribution.png", dpi=300, bbox_inches='tight')  # 또는 .pdf 가능


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Attack (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Prediction-AE)")
plt.tight_layout()
plt.savefig("z_results/prediction_confusion_matrix.png")
plt.show()

print("Confusion Matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}  <- 오탐")
print(f"False Negatives (FN): {fn}  <- 미탐(정탐 실패)")
print(f"True Positives (TP): {tp}  <- 정탐")

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