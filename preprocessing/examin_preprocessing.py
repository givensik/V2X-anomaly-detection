import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# CSV 파일 경로 설정
v2aix_path = 'out/v2aix_preprocessed.csv'
veremi_path = 'out/veremi_preprocessed.csv'

print("=== V2X 전처리 데이터 검사 ===\n")

# 파일 존재 확인
if not os.path.exists(v2aix_path):
    print(f"❌ {v2aix_path} 파일이 존재하지 않습니다.")
    exit()
if not os.path.exists(veremi_path):
    print(f"❌ {veremi_path} 파일이 존재하지 않습니다.")
    exit()

# 데이터 로드
print("📁 데이터 로딩 중...")
v2aix_df = pd.read_csv(v2aix_path)
veremi_df = pd.read_csv(veremi_path)

# 데이터셋 라벨 추가
v2aix_df['dataset'] = 'v2aix'
veremi_df['dataset'] = 'veremi'

print(f"✅ V2AIX 데이터: {len(v2aix_df):,} 행")
print(f"✅ VeReMi 데이터: {len(veremi_df):,} 행")

# 데이터 결합
combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
print(f"✅ 전체 결합 데이터: {len(combined_df):,} 행\n")

# 기본 정보 출력
print("=== 데이터셋 기본 정보 ===")
print("V2AIX 컬럼:", list(v2aix_df.columns))
print("VeReMi 컬럼:", list(veremi_df.columns))
print()

# 공통 컬럼 찾기
common_cols = set(v2aix_df.columns) & set(veremi_df.columns)
print(f"공통 컬럼 ({len(common_cols)}개):", sorted(common_cols))
print()

# 라벨 분포 확인
if 'label' in common_cols:
    print("=== 라벨 분포 ===")
    print("V2AIX 라벨 분포:")
    print(v2aix_df['label'].value_counts())
    print(f"공격 비율: {v2aix_df['label'].mean():.3f}")
    print()
    
    print("VeReMi 라벨 분포:")
    print(veremi_df['label'].value_counts())
    print(f"공격 비율: {veremi_df['label'].mean():.3f}")
    print()

# 주요 피처들의 통계 비교
key_features = ['dpos_x', 'dpos_y', 'dspeed', 'acceleration', 'curvature']
available_features = [col for col in key_features if col in common_cols]

if available_features:
    print("=== 주요 피처 통계 비교 ===")
    for col in available_features:
        print(f"\n📊 {col}:")
        v2aix_stats = v2aix_df[col].describe()
        veremi_stats = veremi_df[col].describe()
        
        print(f"  V2AIX  - 평균: {v2aix_stats['mean']:.4f}, 표준편차: {v2aix_stats['std']:.4f}")
        print(f"  VeReMi - 평균: {veremi_stats['mean']:.4f}, 표준편차: {veremi_stats['std']:.4f}")
        
        # 결측값 확인
        v2aix_null = v2aix_df[col].isnull().sum()
        veremi_null = veremi_df[col].isnull().sum()
        if v2aix_null > 0 or veremi_null > 0:
            print(f"  결측값 - V2AIX: {v2aix_null}, VeReMi: {veremi_null}")

# 시각화
if len(available_features) >= 2:
    print("\n=== 시각화 생성 중 ===")
    
    # 박스플롯으로 분포 비교
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(available_features[:4], 1):  # 최대 4개 피처만
        plt.subplot(2, 2, i)
        sns.boxplot(x='dataset', y=col, data=combined_df)
        plt.title(f"{col} Distribution Comparison")
    
    plt.tight_layout()
    plt.show()
    
    # PCA 분석
    if len(available_features) >= 3:
        print("\n📈 PCA 분석 수행 중...")
        
        # PCA용 피처 선택 (수치형만)
        numeric_features = []
        for col in combined_df.columns:
            if col not in ['dataset', 'label'] and combined_df[col].dtype in ['int64', 'float64']:
                if not combined_df[col].isnull().all():
                    numeric_features.append(col)
        
        if len(numeric_features) >= 2:
            # 결측값 처리
            X = combined_df[numeric_features].fillna(0).values
            X = StandardScaler().fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            combined_df['pca1'] = X_pca[:, 0]
            combined_df['pca2'] = X_pca[:, 1]
            
            plt.figure(figsize=(10, 8))
            # 샘플링해서 시각화 (너무 많으면 느려짐)
            sample_size = min(10000, len(combined_df))
            sample_df = combined_df.sample(sample_size)
            
            sns.scatterplot(x='pca1', y='pca2', hue='dataset', data=sample_df, alpha=0.6)
            plt.title(f"PCA Analysis - Feature Space Comparison\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f})")
            plt.show()
            
            print(f"✅ PCA 완료 - 총 {len(numeric_features)}개 피처 사용")
        else:
            print("❌ PCA 분석을 위한 충분한 수치형 피처가 없습니다.")

print("\n=== 검사 완료 ===")
print("전처리된 데이터의 기본적인 통계와 분포를 확인했습니다.")