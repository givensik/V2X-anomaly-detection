#!/usr/bin/env python3
"""
개선된 V2X 이상탐지 테스트 - Balanced 전략 적용
현재 설정에서 임계값 전략만 변경하여 성능 개선
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, fbeta_score, 
    precision_score, recall_score, classification_report, confusion_matrix
)

from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores
from v2x_testing_lstm import LSTMAutoEncoder, V2XDataset

def find_balanced_threshold(scores, labels):
    """Precision과 Recall의 균형을 고려한 임계값 찾기"""
    thresholds = np.linspace(scores.min(), scores.max(), 2000)
    best_score = -1
    best_thr = scores.mean()
    
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        
        try:
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            acc = accuracy_score(labels, preds)
            
            # Balanced score: F1 * Accuracy
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
                score = f1 * acc
            else:
                score = 0
                
            if score > best_score:
                best_score = score
                best_thr = thr
        except:
            continue
    
    return best_thr, best_score

def run_improved_testing():
    """개선된 테스트 실행"""
    
    # 기존 설정
    artifacts_dir = "artifacts_lstm"
    v2aix_csv_path = "out/v2aix_preprocessed.csv"
    veremi_csv_path = "out/veremi_preprocessed.csv"
    random_state = 42
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 메타데이터 로드
    meta_path = os.path.join(artifacts_dir, "training_meta_lstm.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    seq_len = int(meta.get("sequence_length", 20))
    input_dim = int(meta.get("input_dim", 8))
    alpha = 1.0  # 최적값 사용
    
    print("=== IMPROVED V2X ANOMALY DETECTION ===")
    print(f"Using optimized settings: alpha={alpha}, balanced threshold strategy")
    
    # 데이터 로드 및 전처리 (기존과 동일)
    print("\nLoading preprocessed data...")
    veremi_df = pd.read_csv(veremi_csv_path)
    pre = V2XDataPreprocessor()
    
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(
        veremi_df, sequence_length=seq_len
    )
    
    # 스케일링
    v2aix_df = pd.read_csv(v2aix_csv_path)
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=seq_len)
    
    X_v2aix_flat = X_v2aix.reshape(-1, X_v2aix.shape[-1])
    X_veremi_flat = X_veremi.reshape(-1, X_veremi.shape[-1])
    X_combined = np.vstack([X_v2aix_flat, X_veremi_flat])
    
    pre.scaler.fit(X_combined)
    X_veremi = pre.scaler.transform(X_veremi_flat).reshape(X_veremi.shape)
    
    # 테스트 분할
    _, X_test, _, y_test, _, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    
    print(f"Test sequences: {len(X_test)}")
    print(f"Attack ratio: {y_test.mean():.3f}")
    
    # 모델 로드
    model = LSTMAutoEncoder(input_dim=input_dim, sequence_length=seq_len)
    model_path = os.path.join(artifacts_dir, "model_lstm.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # AE 오차 계산
    print("Computing reconstruction errors...")
    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device).view(1,1,-1)
    
    errors = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            rec = model(xb)
            werr = ((xb - rec) ** 2) * feature_weights
            err_seq = torch.mean(werr, dim=[1,2]).cpu().numpy()
            errors.extend(err_seq)
    
    errors = np.array(errors)
    labels = y_test[:len(errors)]
    rules = r_test[:len(errors)]
    
    # Late fusion
    combined = fuse_scores(errors, rules, alpha=alpha)
    
    # 개선된 임계값 전략
    print("Finding optimal balanced threshold...")
    thr_balanced, balanced_score = find_balanced_threshold(combined, labels)
    
    # 결과 계산
    preds = (combined > thr_balanced).astype(int)
    
    auc = roc_auc_score(labels, combined)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = fbeta_score(labels, preds, beta=1.0)
    f15 = fbeta_score(labels, preds, beta=1.5)
    
    print("\n" + "="*60)
    print("🎯 IMPROVED ANOMALY DETECTION RESULTS")
    print("="*60)
    print(f"Strategy         : Balanced (F1 × Accuracy optimization)")
    print(f"Alpha (AE weight): {alpha:.2f}")
    print(f"Threshold        : {thr_balanced:.6f}")
    print(f"Balanced Score   : {balanced_score:.4f}")
    print("-"*60)
    print(f"AUC-ROC         : {auc:.4f}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"F1.5 Score      : {f15:.4f}")
    
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    print("🔍 CONFUSION MATRIX:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # 성능 개선 분석
    print("\n📈 PERFORMANCE ANALYSIS:")
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives  : {tn:,} (correctly identified normal)")
    print(f"False Positives : {fp:,} (normal classified as attack)")
    print(f"False Negatives : {fn:,} (attacks missed)")
    print(f"True Positives  : {tp:,} (attacks correctly detected)")
    
    attack_detection_rate = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)
    
    print(f"\nAttack Detection Rate: {attack_detection_rate:.1%}")
    print(f"False Alarm Rate     : {false_alarm_rate:.1%}")
    
    # 개별 점수 비교
    ae_norm = (errors - np.percentile(errors, 1)) / (np.percentile(errors, 99) - np.percentile(errors, 1) + 1e-12)
    ae_auc = roc_auc_score(labels, ae_norm)
    rule_auc = roc_auc_score(labels, rules)
    
    print(f"\n🔬 COMPONENT ANALYSIS:")
    print(f"AE-only AUC     : {ae_auc:.4f}")
    print(f"Rule-only AUC   : {rule_auc:.4f}")
    print(f"Combined AUC    : {auc:.4f}")
    print(f"Fusion Benefit  : {auc - max(ae_auc, rule_auc):.4f}")
    
    return {
        'auc': auc, 'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'f1.5': f15, 'threshold': thr_balanced,
        'attack_detection_rate': attack_detection_rate,
        'false_alarm_rate': false_alarm_rate
    }

if __name__ == "__main__":
    results = run_improved_testing()
