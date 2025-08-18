
V2X Anomaly Detection System
이 프로젝트는 V2AIX와 VeReMi 데이터셋을 이용하여 AutoEncoder 기반의 5G/6G V2X(차량-사물 통신) 이상 탐지 시스템을 개발합니다.

개요
V2AIX 데이터셋: 이종 차량의 CAM(협력형 인식 메시지) 데이터로, 모델 학습을 위한 정상 데이터로 활용됩니다.

VeReMi 데이터셋: 공격 메시지가 포함된 CAM 데이터로, 이상(공격) 탐지 성능 평가에 사용됩니다.

AutoEncoder: 정상적인 CAM 메시지 패턴을 학습하여, 정상 패턴에서 벗어나는 이상 메시지를 효과적으로 탐지합니다.

시스템 구조
이 시스템은 세 개의 독립적인 파이썬 스크립트로 구성되어 있습니다.

V2X/
├── v2x_preprocessing.py # 데이터 전처리, 피처 추출, 시퀀스 생성
├── v2x_training.py      # AutoEncoder 모델 학습 및 임계값 설정
├── v2x_testing.py       # 학습된 모델을 이용한 이상 탐지 성능 평가
└── README.md            # 이 파일
주요 기능
1. v2x_preprocessing.py
통합 전처리: V2AIX와 VeReMi 두 데이터셋의 CAM 메시지를 모두 처리합니다.

피처 추출: CAM 메시지에서 위치, 속도, 방향 등 중요한 특성들을 추출합니다.

GroundTruth 활용: VeReMi 데이터셋의 GroundTruth 파일을 이용하여 공격 메시지에 is_attacker 라벨을 부여합니다.

시퀀스 생성: 추출된 데이터를 시간 순서대로 시계열 시퀀스(Sliding Window)로 변환합니다.

2. v2x_training.py
모델 학습: 전처리된 정상 데이터(V2AIX)를 사용하여 AutoEncoder 모델을 학습합니다.

재구성 오차: 학습된 모델의 재구성 오차를 통해 정상 데이터의 패턴을 학습합니다.

임계값 설정: 검증 데이터셋의 재구성 오차 분포를 기반으로 이상 탐지를 위한 임계값을 설정하고 저장합니다.

3. v2x_testing.py
모델 로딩: 학습된 모델과 전처리기, 임계값을 불러옵니다.

성능 평가: 테스트 데이터셋을 사용하여 모델의 이상 탐지 성능(Accuracy, AUC-ROC, Precision, Recall, F1-Score 등)을 평가합니다.

시각화: 재구성 오차 분포 히스토그램과 ROC 곡선 등을 시각화하여 결과를 쉽게 이해할 수 있도록 합니다.

설치 및 실행
1. 필요한 패키지 설치
이 프로젝트를 실행하기 위해 다음 패키지들이 필요합니다. requirements.txt 파일을 만들어 한번에 설치할 수 있습니다.

torch
pandas
numpy
scikit-learn
matplotlib
Bash

# 터미널에서 실행
pip install torch pandas numpy scikit-learn matplotlib
2. 실행 순서
아래 명령어를 순서대로 실행하여 전체 파이프라인을 진행합니다.

데이터 전처리: 원본 데이터를 불러와 전처리하고, out/ 디렉토리에 CSV 파일로 저장합니다.

Bash

python v2x_preprocessing.py
모델 학습: 전처리된 데이터를 이용하여 AutoEncoder 모델을 학습시키고, artifacts/ 디렉토리에 모델, 전처리기, 학습 메타데이터를 저장합니다.

Bash

python v2x_training.py --v2aix_csv_path "out/v2aix_preprocessed.csv" --veremi_csv_path "out/veremi_preprocessed.csv"
모델 테스트: 학습된 모델을 불러와 이상 탐지 성능을 평가합니다.

Bash

python v2x_testing.py --artifacts_dir "artifacts" --v2aix_csv_path "out/v2aix_preprocessed.csv" --veremi_csv_path "out/veremi_preprocessed.csv"
주의사항
v2x_preprocessing.py 실행 시 데이터셋의 경로를 코드 내에서 올바르게 설정해야 합니다.

대용량 데이터를 처리하는 경우 메모리 사용량이 높아질 수 있으니, max_files 또는 batch_size 파라미터를 조정하여 메모리 부족 현상을 방지할 수 있습니다.

GPU(CUDA)를 사용할 수 있는 환경에서는 torch가 자동으로 GPU를 활용하여 학습 속도를 높여줍니다.