# V2X 이상탐지 시스템 성능 평가 가이드

## 📊 주요 성능 지표들

### 1. **기본 분류 지표**
- **AUC-ROC**: 0.7822 (현재 시스템 성능)
- **Accuracy**: 전체 예측 정확도
- **Precision**: 공격으로 예측한 것 중 실제 공격 비율
- **Recall**: 실제 공격 중 올바르게 탐지한 비율
- **F1 Score**: Precision과 Recall의 조화평균
- **F1.5 Score**: Recall에 더 가중치를 둔 지표

### 2. **V2X 특화 지표**
- **Attack Detection Rate**: 73.4% (실제 공격 탐지율)
- **False Alarm Rate**: 28.5% (정상 트래픽 오탐지율)
- **Confusion Matrix**: TP, TN, FP, FN 분석

## 🚀 성능 평가 실행 방법

### 방법 1: 기본 테스트 실행
```bash
python v2x_testing_lstm.py
```

### 방법 2: 개선된 테스트 (권장)
```bash
python run_improved_testing.py
```

### 방법 3: Type 2 공격 전용 평가
```bash
# 1. Type 2 데이터로 전처리
python v2x_preprocessing_lstm.py

# 2. Type 2 전용 모델 훈련
python v2x_training_lstm.py

# 3. Type 2 전용 평가
python run_improved_testing.py
```

## 📈 평가 결과 해석

### 출력 예시:
```
🎯 IMPROVED ANOMALY DETECTION RESULTS
====================================
AUC-ROC         : 0.7822
Accuracy        : 0.7150
Precision       : 0.7340
Recall          : 0.7340
F1 Score        : 0.7340
Attack Detection Rate: 73.4%
False Alarm Rate     : 28.5%
```

### 지표별 의미:
- **AUC > 0.8**: 우수한 성능
- **AUC 0.7-0.8**: 양호한 성능 ✅ (현재 상태)
- **AUC < 0.7**: 개선 필요

- **False Alarm Rate < 20%**: 실용적
- **False Alarm Rate 20-30%**: 개선 여지 있음 ✅ (현재 28.5%)
- **False Alarm Rate > 30%**: 개선 필요

## 🔍 성능 개선 모니터링

### 1. 실시간 성능 확인
```python
# 성능 지표만 빠르게 확인
from run_improved_testing import run_improved_testing
results = run_improved_testing()
print(f"AUC: {results['auc']:.4f}")
print(f"Attack Detection: {results['attack_detection_rate']:.1%}")
print(f"False Alarms: {results['false_alarm_rate']:.1%}")
```

### 2. 공격 타입별 성능 비교
```python
# Type 2 vs 전체 타입 성능 비교를 위해
# directory_filter_types와 data_filter_types를 변경하여
# 각각 전처리 → 훈련 → 평가
```

### 3. 성능 변화 추적
- `artifacts_lstm/training_meta_lstm.json`에 메타데이터 저장
- 각 실험마다 결과를 기록하여 성능 변화 추적

## 📊 시각화 및 분석

### Confusion Matrix 분석:
```
         Predicted
Actual   Normal  Attack
Normal   [[2150   450]]  <- FP: 450 (오탐지)
Attack   [[ 280   820]]  <- FN: 280 (미탐지)
```

### 성능 트레이드오프:
- **Precision ↑**: 오탐지 감소, 하지만 일부 공격 놓칠 수 있음
- **Recall ↑**: 공격 탐지율 증가, 하지만 오탐지도 증가
- **F1 Score**: 두 지표의 균형점

## 🎯 목표 성능 지표

### 현재 성능:
- AUC: 0.7822
- Attack Detection Rate: 73.4%
- False Alarm Rate: 28.5%

### 개선 목표:
- AUC: 0.85+ 
- Attack Detection Rate: 80%+
- False Alarm Rate: 20% 미만

## 🛠️ 성능 개선 방법

1. **공격 타입별 특화**: Type 2만 학습 → 해당 타입 성능 향상
2. **임계값 최적화**: `find_balanced_threshold()` 사용
3. **Feature Engineering**: 새로운 특성 추가
4. **앙상블 방법**: 여러 모델 조합
5. **하이퍼파라미터 튜닝**: alpha, sequence_length 등 최적화
