# V2X Anomaly Detection System

V2AIX와 VeReMi 데이터셋을 이용한 AutoEncoder 기반 V2X 이상탐지 시스템입니다.

## 개요

이 시스템은 다음과 같은 특징을 가집니다:

- **V2AIX 데이터셋**: 정상 CAM 메시지 데이터로 사용
- **VeReMi 데이터셋**: 공격 데이터가 포함된 CAM 메시지로 사용
- **AutoEncoder**: 정상 패턴을 학습하여 이상을 탐지
- **GroundTruth 활용**: VeReMi의 GroundTruth 파일을 이용한 정확한 공격자 라벨링

## 시스템 구조

```
V2X/
├── anomaly_detection_system.py      # 메인 이상탐지 시스템
├── veremi_ground_truth_analyzer.py  # VeReMi GroundTruth 분석 도구
├── requirements.txt                 # 필요한 패키지 목록
└── README.md                       # 이 파일
```

## 주요 기능

### 1. V2XDataPreprocessor
- V2AIX와 VeReMi 데이터셋 모두를 처리할 수 있는 통합 전처리기
- CAM 메시지에서 위치, 속도, 방향 등의 특성 추출
- VeReMi GroundTruth 파일을 이용한 공격자 라벨 매핑
- 시계열 시퀀스 생성

### 2. AutoEncoder
- 정상 데이터만으로 학습
- 재구성 오차를 이용한 이상 탐지
- 다층 신경망 기반 인코더-디코더 구조

### 3. VeReMiGroundTruthAnalyzer
- GroundTruth 파일 분석
- 공격자 타입별 통계
- 시간적/공간적 패턴 분석
- 시각화 및 결과 저장

## 설치 및 실행

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. VeReMi GroundTruth 분석 (선택사항)
```bash
python veremi_ground_truth_analyzer.py
```
- `veremi_analysis/` 디렉토리에 분석 결과가 저장됩니다.

### 3. 이상탐지 시스템 실행
```bash
python anomaly_detection_system.py
```

## 데이터 구조

### V2AIX 데이터
- 경로: `V2AIX_Data/json/Mobile/V2X-only/Aachen/`
- CAM 메시지 구조:
  ```json
  {
    "/v2x/cam": [
      {
        "recording_timestamp_nsec": 1708882004941380742,
        "message": {
          "header": {"station_id": {"value": 1186758095}},
          "cam": {
            "cam_parameters": {
              "basic_container": {
                "reference_position": {
                  "latitude": {"value": 507903158},
                  "longitude": {"value": 60654822}
                }
              },
              "high_frequency_container": {
                "basic_vehicle_container_high_frequency": {
                  "heading": {"heading_value": {"value": 2633}},
                  "speed": {"speed_value": {"value": 211}},
                  "longitudinal_acceleration": {"longitudinal_acceleration_value": {"value": 12}},
                  "curvature": {"curvature_value": {"value": -636}}
                }
              }
            }
          }
        }
      }
    ]
  }
  ```

### VeReMi 데이터
- JSON 로그: `VeReMi_Data/results/JSONlog-*.json`
- GroundTruth: `VeReMi_Data/results/GroundTruthJSONlog.json`
- CAM 메시지 (type=3) 구조:
  ```json
  {
    "type": 3,
    "sendTime": 10800.392893331087,
    "sender": 13,
    "messageID": 38,
    "pos": [3597.1520859538707, 5542.199221013564, 1.895],
    "spd": [-3.178365760312756, 38.7989694164331, 0.0]
  }
  ```

## GroundTruth 활용 방법

VeReMi의 GroundTruth 파일은 다음과 같은 구조를 가집니다:

```json
{
  "type": 4,
  "time": 10800.392784793165,
  "sender": 13,
  "attackerType": 0,  // 0: 정상, >0: 공격자
  "messageID": 38,
  "pos": [3597.1520859538707, 5542.199221013564, 1.895],
  "spd": [-3.178365760312756, 38.7989694164331, 0.0]
}
```

시스템은 다음과 같이 GroundTruth를 활용합니다:

1. **공격자 식별**: `attackerType > 0`인 메시지를 공격으로 분류
2. **매핑**: `(time, sender, messageID)`를 키로 하여 JSON 로그와 GroundTruth를 매핑
3. **라벨링**: 매핑된 메시지에 공격자 라벨 부여

## 특성 추출

시스템에서 추출하는 주요 특성들:

- **위치**: `pos_x`, `pos_y`, `pos_z`
- **속도**: `spd_x`, `spd_y`, `spd_z`, `speed`
- **방향**: `heading`
- **가속도**: `acceleration`
- **곡률**: `curvature`

## 모델 구조

### AutoEncoder 아키텍처
```
Input (flattened sequence) 
    ↓
Encoder: Linear(128) → ReLU → Dropout
    ↓
Encoder: Linear(64) → ReLU → Dropout
    ↓
Encoder: Linear(32) → ReLU → Dropout
    ↓
Decoder: Linear(64) → ReLU → Dropout
    ↓
Decoder: Linear(128) → ReLU → Dropout
    ↓
Output: Linear(input_dim) → Tanh
```

### 학습 과정
1. **정상 데이터만으로 학습**: V2AIX 데이터를 이용하여 정상 패턴 학습
2. **임계값 설정**: 검증 데이터의 재구성 오차 95번째 백분위수를 임계값으로 설정
3. **이상 탐지**: 테스트 데이터의 재구성 오차가 임계값을 초과하면 이상으로 분류

## 결과 해석

### 성능 지표
- **Accuracy**: 전체 정확도
- **AUC-ROC**: ROC 곡선 아래 면적
- **Precision/Recall**: 정밀도와 재현율
- **F1-Score**: 정밀도와 재현율의 조화평균

### 시각화
- **학습 손실**: 에포크별 학습 손실 변화
- **재구성 오차 분포**: 정상/공격 데이터의 재구성 오차 분포
- **ROC 곡선**: 이상 탐지 성능을 나타내는 ROC 곡선

## 사용자 정의

### 특성 선택
```python
# 사용자 정의 특성 리스트
custom_features = ['pos_x', 'pos_y', 'speed', 'heading']
preprocessor = V2XDataPreprocessor(feature_columns=custom_features)
```

### 모델 파라미터 조정
```python
# AutoEncoder 구조 변경
detector = AnomalyDetector(input_dim=input_dim, hidden_dims=[256, 128, 64, 32])

# 학습 파라미터 조정
train_losses = detector.train(train_loader, epochs=100, device='cuda')
```

### 임계값 조정
```python
# 임계값 백분위수 변경 (기본값: 95)
val_errors = detector.compute_threshold(val_loader, percentile=90, device=device)
```

## 주의사항

1. **데이터 크기**: V2AIX 데이터 파일이 매우 클 수 있으므로 `max_files` 파라미터로 제한
2. **메모리 사용량**: 대용량 데이터 처리 시 충분한 메모리 확보 필요
3. **GPU 사용**: CUDA가 설치된 환경에서 GPU 가속 사용 가능
4. **데이터 경로**: 실제 데이터 경로에 맞게 코드 내 경로 수정 필요

## 문제 해결

### 일반적인 오류
1. **파일 경로 오류**: 데이터 파일 경로가 올바른지 확인
2. **메모리 부족**: `max_files` 파라미터를 줄이거나 배치 크기 조정
3. **CUDA 오류**: GPU 메모리 부족 시 `device='cpu'` 사용

### 성능 개선
1. **특성 엔지니어링**: 도메인 지식을 활용한 추가 특성 생성
2. **모델 구조**: 더 깊은 네트워크 또는 다른 아키텍처 시도
3. **데이터 증강**: 정상 데이터의 다양성 증가

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 문의

문제가 있거나 개선 사항이 있으시면 이슈를 등록해 주세요.

