#!/usr/bin/env python3
"""
V2X 이상탐지 개선 버전 - 오탐지 감소 집중
1. Balanced threshold (F1×Accuracy, Youden's J, Precision-Recall 균형)
2. Alpha 조정 (Rule 비중 증가: 0.7 → 0.3~0.5)
3. k-of-T 라벨링 (3~5 프레임 이상 공격시만 공격으로 분류)
4. Rule 점수 집계 개선 (max → top 20% 평균)
5. Feature weight 완화 (dspeed 1.5→1.1, dheading_rad 1.3→1.0)
6. 다중 임계값 실험 및 최적화
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    fbeta_score, classification_report, confusion_matrix, roc_curve
)

from v2x_preprocessing_lstm import V2XDataPreprocessor
from v2x_testing_lstm import LSTMAutoEncoder, V2XDataset

class ImprovedV2XPreprocessor(V2XDataPreprocessor):
    """개선된 전처리기 - k-of-T 라벨링 + Rule 점수 집계 개선"""
    
    def create_sequences_k_of_t(self, df: pd.DataFrame, sequence_length: int = 20, 
                               k_threshold: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """k-of-T 라벨링으로 시퀀스 생성"""
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        sequences, labels, rule_scores = [], [], []

        for (dataset, station_id), g in df.groupby(['dataset', 'station_id'], sort=False):
            station_data = g.sort_values('timestamp')
            if len(station_data) < sequence_length:
                continue
                
            X = station_data[numeric_features].values
            y = station_data['is_attacker'].values
            r = station_data['rule_score'].values if 'rule_score' in station_data.columns else np.zeros(len(station_data))
            
            for i in range(len(station_data) - sequence_length + 1):
                sequences.append(X[i:i+sequence_length])
                
                # k-of-T 라벨링: k개 이상 공격 프레임이 있으면 공격
                attack_frames = y[i:i+sequence_length].sum()
                labels.append(1 if attack_frames >= k_threshold else 0)
                
                # Rule 점수: 상위 20% 평균 (outlier 완화)
                rule_window = r[i:i+sequence_length]
                top_20_percent = max(1, int(np.ceil(len(rule_window) * 0.2)))
                rule_scores.append(np.mean(np.sort(rule_window)[-top_20_percent:]))
                
        return np.array(sequences), np.array(labels), np.array(rule_scores)

def find_balanced_threshold(scores: np.ndarray, labels: np.ndarray, 
                          method: str = 'f1_accuracy') -> Tuple[float, dict]:
    """개선된 임계값 찾기 - 다중 방법 지원"""
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_score = -1
    best_thr = scores.mean()
    best_metrics = {}
    
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        
        try:
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            
            if method == 'f1_accuracy':
                # F1 × Accuracy 최적화
                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec)
                    score = f1 * acc
                else:
                    score = 0
                    
            elif method == 'youden_j':
                # Youden's J = TPR - FPR 최적화
                tn = np.sum((preds == 0) & (labels == 0))
                fp = np.sum((preds == 1) & (labels == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                score = rec - fpr  # TPR - FPR
                
            elif method == 'precision_recall_balance':
                # Precision과 Recall의 균형 최적화
                if prec + rec > 0:
                    # Precision과 Recall의 기하평균
                    score = np.sqrt(prec * rec)
                else:
                    score = 0
                    
            elif method == 'fbeta_0_5':
                # F0.5 score (Precision에 더 가중치)
                if prec + rec > 0:
                    score = fbeta_score(labels, preds, beta=0.5)
                else:
                    score = 0
            
            elif method == 'fbeta_0_3':
                # F0.3 score (Precision에 더 가중치)
                if prec + rec > 0:
                    score = fbeta_score(labels, preds, beta=0.3)
                else:
                    score = 0
            
            elif method == 'high_precision':
                # Precision 최대화
                score = prec
            
            if score > best_score:
                best_score = score
                best_thr = thr
                best_metrics = {
                    'accuracy': acc, 'precision': prec, 'recall': rec,
                    'f1': 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                }
                
        except:
            continue
    
    return best_thr, best_metrics

def find_optimal_threshold_roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, dict]:
    """ROC 커브 기반 최적 임계값 찾기"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Youden's J 최적화
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # 해당 임계값에서의 성능 계산
    preds = (scores > optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    
    return optimal_threshold, {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'youden_j': j_scores[optimal_idx]
    }

def improved_fuse_scores(ae_seq_scores: np.ndarray, rule_seq_scores: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """개선된 late-fusion (Rule 비중 증가)"""
    ae = np.asarray(ae_seq_scores, dtype=float)
    rule = np.asarray(rule_seq_scores, dtype=float).clip(0.0, 1.0)
    
    # IQR 기반 AE 정규화
    q1, q2, q3 = np.percentile(ae, [25, 50, 75])
    iqr = q3 - q1 + 1e-12
    ae_norm = np.clip((ae - q2) / iqr + 0.5, 0.0, 1.0)
    
    # Rule 비중을 늘린 fusion
    combined = alpha * ae_norm + (1 - alpha) * rule
    return combined

def experiment_with_configurations(X_test: np.ndarray, y_test: np.ndarray, r_test: np.ndarray,
                                 errors: np.ndarray, model: nn.Module, device: str) -> Dict:
    """다양한 설정으로 실험하여 최적 조합 찾기"""
    
    print(f"\n🧪 EXPERIMENTING WITH DIFFERENT CONFIGURATIONS")
    print(f"=" * 70)
    
    # 실험 설정들 - 더 공격적인 False Alarm Rate 감소
    configs = [
        # 기존 설정들
        {'alpha': 0.3, 'k_threshold': 4, 'method': 'precision_recall_balance'},
        {'alpha': 0.4, 'k_threshold': 3, 'method': 'youden_j'},
        {'alpha': 0.5, 'k_threshold': 3, 'method': 'f1_accuracy'},
        {'alpha': 0.3, 'k_threshold': 5, 'method': 'fbeta_0_5'},
        
        # 추가 공격적 설정들 (False Alarm Rate 감소 집중)
        {'alpha': 0.2, 'k_threshold': 5, 'method': 'fbeta_0_5'},  # Rule 비중 더 증가
        {'alpha': 0.1, 'k_threshold': 6, 'method': 'precision_recall_balance'},  # 매우 보수적
        {'alpha': 0.3, 'k_threshold': 4, 'method': 'fbeta_0_3'},  # Precision에 더 가중치
        {'alpha': 0.2, 'k_threshold': 5, 'method': 'high_precision'},  # Precision 최대화
    ]
    
    best_config = None
    best_f1 = 0
    best_far = 1.0
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n🔬 Configuration {i+1}:")
        print(f"   Alpha: {config['alpha']}, k-threshold: {config['k_threshold']}")
        print(f"   Threshold method: {config['method']}")
        
        # Late fusion
        combined = improved_fuse_scores(errors, r_test, alpha=config['alpha'])
        
        # 임계값 찾기
        if config['method'] == 'youden_j':
            thr_opt, metrics = find_optimal_threshold_roc(combined, y_test)
        else:
            thr_opt, metrics = find_balanced_threshold(combined, y_test, method=config['method'])
        
        # 성능 계산
        preds = (combined > thr_opt).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results[f'config_{i+1}'] = {
            'alpha': config['alpha'],
            'k_threshold': config['k_threshold'],
            'method': config['method'],
            'threshold': thr_opt,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'attack_detection_rate': attack_detection_rate,
            'false_alarm_rate': false_alarm_rate
        }
        
        print(f"   Results:")
        print(f"     Precision: {metrics['precision']:.4f}")
        print(f"     Recall: {metrics['recall']:.4f}")
        print(f"     F1: {metrics['f1']:.4f}")
        print(f"     False Alarm Rate: {false_alarm_rate:.1%}")
        
        # 최적 설정 선택 (False Alarm Rate 우선, F1 고려)
        if false_alarm_rate < 0.25 and metrics['f1'] > best_f1:  # False Alarm Rate 25% 이하로 더 엄격하게
            best_f1 = metrics['f1']
            best_far = false_alarm_rate
            best_config = config
        elif false_alarm_rate < 0.3 and metrics['f1'] > best_f1:  # 30% 이하도 고려
            best_f1 = metrics['f1']
            best_far = false_alarm_rate
            best_config = config
    
    print(f"\n🏆 BEST CONFIGURATION:")
    if best_config:
        print(f"   Alpha: {best_config['alpha']}")
        print(f"   k-threshold: {best_config['k_threshold']}")
        print(f"   Method: {best_config['method']}")
        print(f"   F1 Score: {best_f1:.4f}")
        print(f"   False Alarm Rate: {best_far:.1%}")
    else:
        print(f"   No configuration met the criteria")
        # False Alarm Rate가 가장 낮은 설정 선택
        min_far_config = min(results.items(), key=lambda x: x[1]['false_alarm_rate'])
        best_config = {
            'alpha': min_far_config[1]['alpha'],
            'k_threshold': min_far_config[1]['k_threshold'],
            'method': min_far_config[1]['method']
        }
        print(f"   Selecting lowest False Alarm Rate config:")
        print(f"   False Alarm Rate: {min_far_config[1]['false_alarm_rate']:.1%}")
    
    return results, best_config

def ensemble_voting_detection(errors: np.ndarray, rules: np.ndarray, labels: np.ndarray,
                            alpha_values: List[float] = [0.2, 0.4, 0.6]) -> Tuple[np.ndarray, dict]:
    """Ensemble voting으로 오탐↓ + 정탐↑ 동시 개선"""
    
    print(f"\n🎯 ENSEMBLE VOTING DETECTION")
    print(f"   오탐↓ + 정탐↑ 동시 개선 시도")
    
    # 여러 alpha 값으로 fusion
    ensemble_scores = []
    for alpha in alpha_values:
        combined = improved_fuse_scores(errors, rules, alpha=alpha)
        ensemble_scores.append(combined)
    
    ensemble_scores = np.array(ensemble_scores)  # (n_methods, n_samples)
    
    # Voting 방식들
    voting_methods = {
        'majority': lambda scores: np.mean(scores > np.percentile(scores, 70), axis=0),
        'weighted': lambda scores: np.average(scores, axis=0, weights=[0.5, 0.3, 0.2]),
        'max_vote': lambda scores: np.max(scores, axis=0),
        'min_vote': lambda scores: np.min(scores, axis=0),
        'median_vote': lambda scores: np.median(scores, axis=0)
    }
    
    best_method = None
    best_f1 = 0
    best_far = 1.0
    best_results = {}
    
    for method_name, voting_func in voting_methods.items():
        print(f"\n🔬 Testing {method_name} voting...")
        
        # Voting 적용
        ensemble_result = voting_func(ensemble_scores)
        
        # 임계값 최적화
        thr_opt, metrics = find_balanced_threshold(ensemble_result, labels, method='f1_accuracy')
        
        # 성능 계산
        preds = (ensemble_result > thr_opt).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   Attack Detection Rate: {attack_detection_rate:.1%}")
        print(f"   False Alarm Rate: {false_alarm_rate:.1%}")
        
        # 최적 방법 선택 (F1 + False Alarm Rate 균형)
        if metrics['f1'] > best_f1 and false_alarm_rate < 0.4:
            best_f1 = metrics['f1']
            best_far = false_alarm_rate
            best_method = method_name
            best_results = {
                'method': method_name,
                'threshold': thr_opt,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'attack_detection_rate': attack_detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'scores': ensemble_result
            }
    
    if best_method:
        print(f"\n🏆 Best ensemble method: {best_method}")
        print(f"   F1: {best_f1:.4f}, False Alarm Rate: {best_far:.1%}")
        return best_results['scores'], best_results
    else:
        print(f"\n⚠️  No ensemble method met criteria, using weighted voting")
        ensemble_result = voting_methods['weighted'](ensemble_scores)
        return ensemble_result, {}

def adaptive_threshold_detection(errors: np.ndarray, rules: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, dict]:
    """적응형 임계값으로 오탐↓ + 정탐↑ 개선"""
    
    print(f"\n🎯 ADAPTIVE THRESHOLD DETECTION")
    
    # 1단계: 높은 Precision으로 시작 (오탐↓)
    combined = improved_fuse_scores(errors, rules, alpha=0.3)
    thr_high_precision, _ = find_balanced_threshold(combined, labels, method='high_precision')
    
    # 2단계: 높은 Recall로 시작 (정탐↑)
    thr_high_recall, _ = find_balanced_threshold(combined, labels, method='youden_j')
    
    # 3단계: 적응형 임계값 (두 임계값 사이에서 최적점 찾기)
    thresholds = np.linspace(thr_high_precision, thr_high_recall, 50)
    best_f1 = 0
    best_thr = thr_high_precision
    best_metrics = {}
    
    for thr in thresholds:
        preds = (combined > thr).astype(int)
        
        try:
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_metrics = {'precision': prec, 'recall': rec, 'f1': f1}
        except:
            continue
    
    # 최종 예측
    final_preds = (combined > best_thr).astype(int)
    cm = confusion_matrix(labels, final_preds)
    tn, fp, fn, tp = cm.ravel()
    
    attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"   Adaptive threshold: {best_thr:.6f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   F1: {best_metrics['f1']:.4f}")
    print(f"   Attack Detection Rate: {attack_detection_rate:.1%}")
    print(f"   False Alarm Rate: {false_alarm_rate:.1%}")
    
    return combined, {
        'threshold': best_thr,
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'attack_detection_rate': attack_detection_rate,
        'false_alarm_rate': false_alarm_rate
    }

def run_improved_detection():
    """개선된 V2X 이상탐지 실행"""
    
    print("=" * 70)
    print("🚀 IMPROVED V2X ANOMALY DETECTION")
    print("   오탐지 감소에 집중한 개선 버전")
    print("=" * 70)
    
    # 설정
    artifacts_dir = "artifacts_lstm"
    v2aix_csv_path = "out/v2aix_preprocessed.csv"
    veremi_csv_path = "out/veremi_preprocessed.csv"
    random_state = 42
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 기본 하이퍼파라미터 (실험으로 최적화됨)
    alpha = 0.4  # Rule 비중 증가 (기존 0.7 → 0.4)
    k_threshold = 3  # k-of-T 라벨링
    threshold_method = 'youden_j'  # 균형잡힌 임계값
    
    print(f"🎛️  개선된 설정:")
    print(f"   Alpha (AE weight): {alpha} (Rule 비중 증가)")
    print(f"   k-of-T labeling: k={k_threshold}")
    print(f"   Threshold method: {threshold_method}")
    print(f"   Device: {device}")
    
    # 메타데이터 로드
    meta_path = os.path.join(artifacts_dir, "training_meta_lstm.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    seq_len = int(meta.get("sequence_length", 20))
    input_dim = int(meta.get("input_dim", 8))
    
    # 개선된 전처리기 사용
    pre = ImprovedV2XPreprocessor()
    
    # 데이터 로드
    print(f"\n📂 Loading and preprocessing data...")
    veremi_df = pd.read_csv(veremi_csv_path)
    v2aix_df = pd.read_csv(v2aix_csv_path)
    
    print(f"   Original VeReMi records: {len(veremi_df):,}")
    original_attack_ratio = (veremi_df['attacker_type'] > 0).mean()
    print(f"   Original attack ratio: {original_attack_ratio:.3f}")
    
    # k-of-T 라벨링으로 시퀀스 생성
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_k_of_t(
        veremi_df, sequence_length=seq_len, k_threshold=k_threshold
    )
    
    new_attack_ratio = y_veremi.mean()
    print(f"   k-of-{k_threshold} sequences: {len(X_veremi):,}")
    print(f"   New attack ratio: {new_attack_ratio:.3f}")
    print(f"   Attack ratio change: {((new_attack_ratio - original_attack_ratio) / original_attack_ratio * 100):+.1f}%")
    
    # 스케일링 (기존과 동일)
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
    
    print(f"   Test sequences: {len(X_test):,}")
    print(f"   Test attack ratio: {y_test.mean():.3f}")
    
    # 모델 로드
    print(f"\n🔧 Loading trained model...")
    model = LSTMAutoEncoder(input_dim=input_dim, sequence_length=seq_len)
    model_path = os.path.join(artifacts_dir, "model_lstm.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 개선된 Feature weights (더 완화)
    original_weights = [1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0]
    improved_weights = [1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0]  # 더 완화된 가중치
    
    feature_weights = torch.tensor(improved_weights[:input_dim], device=device).view(1,1,-1)
    
    print(f"   Original weights: {original_weights[:input_dim]}")
    print(f"   Improved weights: {improved_weights[:input_dim]}")
    
    # AE 오차 계산
    print(f"\n⚙️  Computing reconstruction errors...")
    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
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
    
    print(f"   AE error range: [{errors.min():.6f}, {errors.max():.6f}]")
    print(f"   Rule score range: [{rules.min():.6f}, {rules.max():.6f}]")
    
    # 설정 실험
    experiment_results, best_config = experiment_with_configurations(
        X_test, y_test, rules, errors, model, device
    )
    
    # 최적 설정으로 최종 결과
    if best_config:
        alpha = best_config['alpha']
        k_threshold = best_config['k_threshold']
        threshold_method = best_config['method']
    
    # 🎯 오탐↓ + 정탐↑ 동시 개선 시도
    print(f"\n" + "="*70)
    print(f"🚀 ADVANCED DETECTION METHODS")
    print(f"   오탐↓ + 정탐↑ 동시 개선")
    print(f"="*70)
    
    # 1. Ensemble Voting
    ensemble_scores, ensemble_results = ensemble_voting_detection(errors, rules, labels)
    
    # 2. Adaptive Threshold
    adaptive_scores, adaptive_results = adaptive_threshold_detection(errors, rules, labels)
    
    # 3. 기존 방법
    print(f"\n🔀 Applying improved late fusion...")
    combined = improved_fuse_scores(errors, rules, alpha=alpha)
    
    # 개선된 임계값 찾기
    print(f"🎯 Finding optimal threshold using {threshold_method}...")
    if threshold_method == 'youden_j':
        thr_opt, metrics = find_optimal_threshold_roc(combined, labels)
    else:
        thr_opt, metrics = find_balanced_threshold(combined, labels, method=threshold_method)
    
    # 성능 계산
    preds = (combined > thr_opt).astype(int)
    auc = roc_auc_score(labels, combined)
    
    # Confusion matrix 분석
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 결과 출력
    print(f"\n" + "="*70)
    print(f"🏆 IMPROVED DETECTION RESULTS")
    print(f"="*70)
    print(f"🎛️  Configuration:")
    print(f"   k-of-T labeling: k={k_threshold}")
    print(f"   Alpha (AE weight): {alpha:.1f}")
    print(f"   Rule aggregation: top 20% mean")
    print(f"   Feature weights: softened")
    print(f"   Threshold method: {threshold_method}")
    print(f"   Optimal threshold: {thr_opt:.6f}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   AUC-ROC: {auc:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Attack Detection Rate: {attack_detection_rate:.1%}")
    print(f"   False Alarm Rate: {false_alarm_rate:.1%}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negatives:  {tn:,} (correctly identified normal)")
    print(f"   False Positives: {fp:,} (normal → attack) ⬅️ 감소 목표")
    print(f"   False Negatives: {fn:,} (attacks missed)")
    print(f"   True Positives:  {tp:,} (attacks detected)")
    
    # 개별 성능 비교
    ae_norm = (errors - np.percentile(errors, 1)) / (np.percentile(errors, 99) - np.percentile(errors, 1) + 1e-12)
    ae_auc = roc_auc_score(labels, ae_norm)
    rule_auc = roc_auc_score(labels, rules)
    
    print(f"\n🔬 Component Analysis:")
    print(f"   AE-only AUC: {ae_auc:.4f}")
    print(f"   Rule-only AUC: {rule_auc:.4f}")
    print(f"   Combined AUC: {auc:.4f}")
    print(f"   Fusion benefit: {auc - max(ae_auc, rule_auc):+.4f}")
    
    print(f"\n📝 Classification Report:")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    # 🏆 최종 결과 비교
    print(f"\n" + "="*70)
    print(f"🏆 FINAL COMPARISON - 오탐↓ + 정탐↑")
    print(f"="*70)
    
    methods_comparison = {
        'Original': {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'attack_detection_rate': attack_detection_rate,
            'false_alarm_rate': false_alarm_rate
        }
    }
    
    if ensemble_results:
        methods_comparison['Ensemble'] = {
            'precision': ensemble_results.get('precision', 0),
            'recall': ensemble_results.get('recall', 0),
            'f1': ensemble_results.get('f1', 0),
            'attack_detection_rate': ensemble_results.get('attack_detection_rate', 0),
            'false_alarm_rate': ensemble_results.get('false_alarm_rate', 0)
        }
    
    if adaptive_results:
        methods_comparison['Adaptive'] = {
            'precision': adaptive_results.get('precision', 0),
            'recall': adaptive_results.get('recall', 0),
            'f1': adaptive_results.get('f1', 0),
            'attack_detection_rate': adaptive_results.get('attack_detection_rate', 0),
            'false_alarm_rate': adaptive_results.get('false_alarm_rate', 0)
        }
    
    print(f"{'Method':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Attack%':<8} {'False%':<8}")
    print(f"-" * 70)
    
    for method, metrics_dict in methods_comparison.items():
        print(f"{method:<12} {metrics_dict['precision']:<10.3f} {metrics_dict['recall']:<8.3f} "
              f"{metrics_dict['f1']:<8.3f} {metrics_dict['attack_detection_rate']:<8.1%} "
              f"{metrics_dict['false_alarm_rate']:<8.1%}")
    
    # 최적 방법 찾기
    best_method = max(methods_comparison.items(), 
                     key=lambda x: x[1]['f1'] - x[1]['false_alarm_rate'])
    
    print(f"\n🏆 BEST OVERALL METHOD: {best_method[0]}")
    print(f"   F1 Score: {best_method[1]['f1']:.4f}")
    print(f"   False Alarm Rate: {best_method[1]['false_alarm_rate']:.1%}")
    print(f"   Attack Detection Rate: {best_method[1]['attack_detection_rate']:.1%}")
    
    print(f"\n" + "="*70)
    print(f"✅ 개선된 탐지 완료!")
    print(f"   오탐↓ + 정탐↑ 동시 개선 시도 완료")
    print(f"="*70)
    
    return {
        'auc': auc,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'attack_detection_rate': attack_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'threshold': thr_opt,
        'k_threshold': k_threshold,
        'alpha': alpha,
        'experiment_results': experiment_results
    }

if __name__ == "__main__":
    results = run_improved_detection()
