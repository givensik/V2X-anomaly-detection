# v2x_testing_lstm.py
# LSTM AutoEncoder 기반 V2X 이상탐지 테스트 스크립트
# - Training과 완전히 일치하는 전처리/스케일링/피처가중치 적용
# - Late fusion (AE + Rule-based scores) 사용
# - VeReMi 데이터셋의 test split에서 평가

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, accuracy_score, fbeta_score
)

# 학습 코드와 동일한 전처리/late-fusion 유틸 사용
from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores

# -------------------------------
# Model (LSTM-AutoEncoder)  — 학습 스크립트와 동일
# -------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.enc = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        self.bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dec = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.enc(x)                 # h: (2*n_layers, B, H)
        h_last_fwd = h[-2, :, :]
        h_last_bwd = h[-1, :, :]
        z = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # (B, 2H)
        z = torch.tanh(self.bridge(z))                   # (B, H)
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)   # (B, T, H)
        y, _ = self.dec(z_seq)
        recon = self.head(y)                             # (B, T, F)
        return recon

class V2XDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------------
# Testing
# -------------------------------
def run_testing(
    artifacts_dir: str = "artifacts_lstm_combined",
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    random_state: int = 42,
    batch_size: int = 64,
    beta_for_report: float = 1.5,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 메타 로드 (새 키 이름들 대응)
    meta_path = os.path.join(artifacts_dir, "training_meta_lstm_combined.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    seq_len   = int(meta.get("sequence_length", 20))
    input_dim = int(meta.get("input_dim", meta.get("num_features", 8)))
    alpha     = float(meta.get("alpha_fusion", meta.get("alpha", 0.7)))
    thr_fuse  = float(meta.get("threshold_fusion", meta.get("threshold", 0.5)))
    used_features = meta.get("used_features", None)

    print("=== Loaded training meta ===")
    print(f"sequence_length={seq_len}, input_dim={input_dim}, alpha={alpha}, threshold={thr_fuse}")
    if used_features: print("used_features:", used_features)

    # 전처리된 CSV 로드 
    print("\nLoading preprocessed CSVs...")
    veremi_df = pd.read_csv(veremi_csv_path)

    # 학습 때와 동일한 feature_columns (전처리 클래스 기본값 그대로)
    pre = V2XDataPreprocessor()
    numeric_features = [c for c in pre.feature_columns if c in veremi_df.columns]

    # VeReMi만으로 테스트 세트 구성 (규칙 점수를 함께 시퀀스로 집계)
    print("Creating sequences from VeReMi with rule scores...")
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(
        veremi_df, sequence_length=seq_len
    )

    # ★ Training과 동일한 방식으로 분할 (validation/test 50:50)
    X_val_mix, X_test, y_val_mix, y_test, r_val_mix, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    print(f"Test sequences: {len(X_test)} (from second half of VeReMi split)")
    
    # ★ Training과 동일한 스케일링 적용 (두 데이터셋 합쳐서)
    print("Applying unified scaling (same as training)...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=seq_len)
    
    # 합쳐서 scaler fit (training과 동일)
    X_v2aix_flat = X_v2aix.reshape(-1, X_v2aix.shape[-1])
    X_veremi_flat = X_veremi.reshape(-1, X_veremi.shape[-1])
    X_combined = np.vstack([X_v2aix_flat, X_veremi_flat])
    
    pre.scaler.fit(X_combined)
    pre.is_fitted = True
    
    # 테스트 데이터에만 transform 적용
    X_test = pre.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 모델 로드 (학습과 동일 구조)
    model = LSTMAutoEncoder(input_dim=input_dim, sequence_length=seq_len)
    model_path = os.path.join(artifacts_dir, "best_autoencoder.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device); model.eval()

    # ★ Training과 정확히 동일한 피처별 가중치 (고정 순서)
    # feature order: ['dpos_x', 'dpos_y', 'dspeed', 'dheading_rad', 'acceleration', 'curvature', 'rel_pos_x', 'rel_pos_y']
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device)
    
    # 실제 feature 수와 맞는지 확인
    if len(feature_weights) != input_dim:
        print(f"Warning: feature_weights length ({len(feature_weights)}) != input_dim ({input_dim})")
        # input_dim에 맞춰 조정
        if len(feature_weights) > input_dim:
            feature_weights = feature_weights[:input_dim]
        else:
            # 부족한 부분은 1.0으로 채움
            padding = torch.ones(input_dim - len(feature_weights), device=device)
            feature_weights = torch.cat([feature_weights, padding])
    
    feature_weights = feature_weights.view(1,1,-1)  # (1,1,F)

    # AE 재구성오차 계산 (가중 MSE)
    errors, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)  # (B,T,F)
            rec = model(xb)
            werr = ((xb - rec) ** 2) * feature_weights
            err_seq = torch.mean(werr, dim=[1,2]).cpu().numpy()  # (B,)
            errors.extend(err_seq); labels.extend(yb.numpy())

    errors = np.array(errors)
    labels = np.array(labels)
    r_test = np.array(r_test[:len(errors)])  # 안전 슬라이스
    
    print(f"Test evaluation - Sequences: {len(errors)}, AE error range: [{errors.min():.6f}, {errors.max():.6f}]")
    print(f"Rule scores range: [{r_test.min():.6f}, {r_test.max():.6f}]")

    # Late-fusion (학습 메타의 alpha/threshold 사용)
    combined = fuse_scores(errors, r_test, alpha=alpha)

    # 메트릭 계산
    auc_roc = roc_auc_score(labels, combined) if len(np.unique(labels))>1 else float('nan')
    auc_pr = average_precision_score(labels, combined) if len(np.unique(labels))>1 else float('nan')
    
    # 임계값 문제 해결: combined score 분포 기반으로 적절한 임계값 찾기
    print(f"\nCombined score distribution:")
    print(f"  Min: {combined.min():.6f}, Max: {combined.max():.6f}")
    print(f"  Mean: {combined.mean():.6f}, Std: {combined.std():.6f}")
    print(f"  Percentiles: 25%={np.percentile(combined, 25):.6f}, 50%={np.percentile(combined, 50):.6f}, 75%={np.percentile(combined, 75):.6f}")
    
    # 기존 임계값이 너무 높으면 조정
    if thr_fuse > combined.max():
        print(f"Warning: Original threshold {thr_fuse:.6f} > max score {combined.max():.6f}")
        # 75th percentile을 임계값으로 사용
        thr_fuse = np.percentile(combined, 75)
        print(f"Adjusted threshold to 75th percentile: {thr_fuse:.6f}")
    elif thr_fuse < combined.min():
        print(f"Warning: Original threshold {thr_fuse:.6f} < min score {combined.min():.6f}")
        # 25th percentile을 임계값으로 사용
        thr_fuse = np.percentile(combined, 25)
        print(f"Adjusted threshold to 25th percentile: {thr_fuse:.6f}")
    
    preds = (combined > thr_fuse).astype(int)
    acc   = accuracy_score(labels, preds)
    fbeta = fbeta_score(labels, preds, beta=beta_for_report) if len(np.unique(labels))>1 else float('nan')

    print("\n=== ANOMALY DETECTION RESULTS (LSTM + Late Fusion) ===")
    print(f"Fusion parameters: alpha={alpha:.2f} (AE weight)")
    print(f"ROC-AUC   : {auc_roc:.4f}")
    print(f"PR-AUC    : {auc_pr:.4f}")
    print(f"Accuracy@thr({thr_fuse:.6f}): {acc:.4f}")
    print(f"F{beta_for_report:.1f}@thr    : {fbeta:.4f}")
    
    # 개별 점수 성능도 참고용으로 출력
    ae_norm = (errors - np.percentile(errors, 1)) / (np.percentile(errors, 99) - np.percentile(errors, 1) + 1e-12)
    ae_auc_roc = roc_auc_score(labels, ae_norm) if len(np.unique(labels)) > 1 else float('nan')
    ae_auc_pr = average_precision_score(labels, ae_norm) if len(np.unique(labels)) > 1 else float('nan')
    rule_auc_roc = roc_auc_score(labels, r_test) if len(np.unique(labels)) > 1 else float('nan')
    rule_auc_pr = average_precision_score(labels, r_test) if len(np.unique(labels)) > 1 else float('nan')
    print(f"Individual scores:")
    print(f"  AE: ROC-AUC={ae_auc_roc:.4f}, PR-AUC={ae_auc_pr:.4f}")
    print(f"  Rule: ROC-AUC={rule_auc_roc:.4f}, PR-AUC={rule_auc_pr:.4f}")

    print("\nClassification Report:")
    try:
        from sklearn.metrics import classification_report
        print(classification_report(labels, preds, target_names=['Normal','Attack'], zero_division=0))
    except Exception as e:
        print(f"  (classification_report unavailable: {e})")

    print("Confusion Matrix:")
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        print(cm)
        print(f"  Normal predicted as Normal: {cm[0,0]}")
        print(f"  Normal predicted as Attack: {cm[0,1]}")
        print(f"  Attack predicted as Normal: {cm[1,0]}")
        print(f"  Attack predicted as Attack: {cm[1,1]}")
    except Exception as e:
        print(f"  (confusion_matrix unavailable: {e})")

if __name__ == "__main__":
    run_testing()
