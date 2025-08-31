# v2x_testing_optimized.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores
from v2x_training_optimized import LSTMAutoEncoder, V2XDataset

def run_testing_optimized(
    artifacts_dir: str = "artifacts_lstm_optimized",
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    random_state: int = 42,
    batch_size: int = 64,
):
    """최적화된 설정으로 테스트 실행"""
    
    print("�� OPTIMIZED V2X TESTING - Alpha=1.0, Balanced Strategy")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 메타 로드
    meta_path = os.path.join(artifacts_dir, "training_meta_lstm_optimized.json")
    
    if not os.path.exists(meta_path):
        print(f"❌ Optimized meta file not found: {meta_path}")
        print("Please run optimized training first!")
        return None
        
    with open(meta_path, "r") as f:
        meta = json.load(f)

    seq_len = int(meta.get("sequence_length", 20))
    input_dim = int(meta.get("input_dim", 8))
    alpha = float(meta.get("alpha_fusion", 1.0))
    thr_fuse = float(meta.get("threshold_fusion", 0.5))
    strategy = meta.get("strategy", "balanced")
    used_features = meta.get("used_features", None)

    print(f"Loaded optimized settings:")
    print(f"  Alpha: {alpha} (AE weight)")
    print(f"  Strategy: {strategy}")
    print(f"  Threshold: {thr_fuse:.6f}")
    print(f"  Features: {len(used_features) if used_features else 'default'}")

    # 데이터 로드 및 전처리 (training과 동일)
    veremi_df = pd.read_csv(veremi_csv_path)
    pre = V2XDataPreprocessor()

    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(
        veremi_df, sequence_length=seq_len
    )

    # 스케일링 (training과 동일)
    v2aix_df = pd.read_csv(v2aix_csv_path)
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=seq_len)
    
    X_v2aix_flat = X_v2aix.reshape(-1, X_v2aix.shape[-1])
    X_veremi_flat = X_veremi.reshape(-1, X_veremi.shape[-1])
    X_combined = np.vstack([X_v2aix_flat, X_veremi_flat])
    
    pre.scaler.fit(X_combined)
    X_veremi = pre.scaler.transform(X_veremi_flat).reshape(X_veremi.shape)

    # 테스트 분할 (training과 동일)
    _, X_test, _, y_test, _, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )

    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 최적화된 모델 로드
    model = LSTMAutoEncoder(input_dim=input_dim, sequence_length=seq_len)
    model_path = os.path.join(artifacts_dir, "model_lstm_optimized.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ Optimized model not found: {model_path}")
        print("Please run optimized training first!")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # AE 재구성오차 계산 (training과 동일)
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device).view(1,1,-1)
    
    errors = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            rec = model(xb)
            werr = ((xb - rec) ** 2) * feature_weights
            err_seq = torch.mean(werr, dim=[1,2]).cpu().numpy()
            errors.extend(err_seq)

    errors = np.array(errors)
    labels = y_test[:len(errors)]
    rules = r_test[:len(errors)]

    # 최적화된 Late-fusion
    combined = fuse_scores(errors, rules, alpha=alpha)

    # 최적화된 임계값으로 평가
    preds = (combined > thr_fuse).astype(int)
    
    auc = roc_auc_score(labels, combined) if len(np.unique(labels)) > 1 else float('nan')
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    balanced_score = f1 * acc

    print(f"\n🏆 OPTIMIZED TEST RESULTS:")
    print(f"AUC-ROC     : {auc:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Balanced Score: {balanced_score:.4f}")

    # 개별 점수 비교
    ae_norm = (errors - np.percentile(errors, 1)) / (np.percentile(errors, 99) - np.percentile(errors, 1) + 1e-12)
    ae_auc = roc_auc_score(labels, ae_norm) if len(np.unique(labels)) > 1 else float('nan')
    rule_auc = roc_auc_score(labels, rules) if len(np.unique(labels)) > 1 else float('nan')

    print(f"\n�� Component Analysis:")
    print(f"AE-only AUC    : {ae_auc:.4f}")
    print(f"Rule-only AUC  : {rule_auc:.4f}")
    print(f"Combined AUC   : {auc:.4f}")
    print(f"Fusion Benefit : {auc - max(ae_auc, rule_auc):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n📈 Classification Report:")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    print(f"Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    attack_detection_rate = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)
    
    print(f"\n�� Key Metrics:")
    print(f"Attack Detection Rate: {attack_detection_rate:.1%}")
    print(f"False Alarm Rate     : {false_alarm_rate:.1%}")
    
    return {
        'auc': auc, 'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'balanced_score': balanced_score,
        'attack_detection_rate': attack_detection_rate,
        'false_alarm_rate': false_alarm_rate
    }

if __name__ == "__main__":
    results = run_testing_optimized()
    if results:
        print(f"\n✅ Optimized testing completed!")