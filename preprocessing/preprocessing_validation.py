import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_validate_data():
    """데이터 로딩 및 기본 검증"""
    print("=== V2X 전처리 데이터 품질 검증 ===\n")
    
    # CSV 파일 경로 설정
    v2aix_path = 'out/v2aix_preprocessed_fixed.csv'
    veremi_path = 'out/veremi_preprocessed_fixed.csv'
    
    # 파일 존재 확인
    if not os.path.exists(v2aix_path):
        print(f"❌ {v2aix_path} 파일이 존재하지 않습니다.")
        return None, None
    if not os.path.exists(veremi_path):
        print(f"❌ {veremi_path} 파일이 존재하지 않습니다.")
        return None, None
    
    # 데이터 로드
    print("📁 데이터 로딩 중...")
    v2aix_df = pd.read_csv(v2aix_path)
    veremi_df = pd.read_csv(veremi_path)
    
    # 데이터셋 라벨 추가
    v2aix_df['dataset'] = 'V2AIX'
    veremi_df['dataset'] = 'VeReMi'
    
    print(f"✅ V2AIX 데이터: {len(v2aix_df):,} 행, {len(v2aix_df.columns)} 컬럼")
    print(f"✅ VeReMi 데이터: {len(veremi_df):,} 행, {len(veremi_df.columns)} 컬럼")
    
    return v2aix_df, veremi_df

def analyze_data_quality(v2aix_df, veremi_df):
    """데이터 품질 분석"""
    print("\n=== 데이터 품질 분석 ===")
    
    # 결측값 분석
    print("\n📊 결측값 분석:")
    for name, df in [("V2AIX", v2aix_df), ("VeReMi", veremi_df)]:
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percent': missing_percent
        }).sort_values('Missing Count', ascending=False)
        
        print(f"\n{name} 결측값:")
        print(missing_info[missing_info['Missing Count'] > 0])
    
    # 중복 데이터 확인
    print(f"\n🔄 중복 데이터:")
    print(f"V2AIX 중복: {v2aix_df.duplicated().sum():,} 행")
    print(f"VeReMi 중복: {veremi_df.duplicated().sum():,} 행")
    
    # 데이터 타입 확인
    print(f"\n📋 데이터 타입:")
    for name, df in [("V2AIX", v2aix_df), ("VeReMi", veremi_df)]:
        print(f"\n{name} 데이터 타입:")
        print(df.dtypes.value_counts())

def analyze_label_distribution(v2aix_df, veremi_df):
    """라벨 분포 분석"""
    print("\n=== 라벨 분포 분석 ===")
    
    if 'label' not in v2aix_df.columns or 'label' not in veremi_df.columns:
        print("❌ 'label' 컬럼이 없습니다.")
        return
    
    # 라벨 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (name, df) in enumerate([("V2AIX", v2aix_df), ("VeReMi", veremi_df)]):
        label_counts = df['label'].value_counts()
        attack_ratio = df['label'].mean()
        
        print(f"\n{name}:")
        print(f"  정상: {label_counts.get(0, 0):,} ({1-attack_ratio:.3f})")
        print(f"  공격: {label_counts.get(1, 0):,} ({attack_ratio:.3f})")
        
        # 파이 차트
        axes[i].pie(label_counts.values, labels=['Normal', 'Attack'], autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'{name} Label Distribution')
    
    plt.tight_layout()
    plt.show()

def analyze_feature_statistics(v2aix_df, veremi_df):
    """피처 통계 분석"""
    print("\n=== 피처 통계 분석 ===")
    
    # 공통 컬럼 찾기
    common_cols = set(v2aix_df.columns) & set(veremi_df.columns)
    numeric_cols = []
    
    for col in common_cols:
        if col not in ['dataset', 'label'] and v2aix_df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
    
    print(f"분석할 수치형 피처: {len(numeric_cols)}개")
    
    if len(numeric_cols) == 0:
        print("❌ 분석할 수치형 피처가 없습니다.")
        return
    
    # 통계 비교 테이블
    stats_comparison = []
    for col in numeric_cols[:10]:  # 상위 10개만
        v2aix_stats = v2aix_df[col].describe()
        veremi_stats = veremi_df[col].describe()
        
        stats_comparison.append({
            'Feature': col,
            'V2AIX_Mean': v2aix_stats['mean'],
            'V2AIX_Std': v2aix_stats['std'],
            'VeReMi_Mean': veremi_stats['mean'],
            'VeReMi_Std': veremi_stats['std'],
            'Mean_Diff': abs(v2aix_stats['mean'] - veremi_stats['mean']),
            'Std_Diff': abs(v2aix_stats['std'] - veremi_stats['std'])
        })
    
    stats_df = pd.DataFrame(stats_comparison)
    print("\n📊 주요 피처 통계 비교:")
    print(stats_df.round(4))
    
    return numeric_cols

def visualize_feature_distributions(v2aix_df, veremi_df, numeric_cols):
    """피처 분포 시각화"""
    print("\n=== 피처 분포 시각화 ===")
    
    if len(numeric_cols) == 0:
        return
    
    # 데이터 결합
    combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    
    # 상위 8개 피처만 시각화
    top_features = numeric_cols[:8]
    
    # 박스플롯
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        sns.boxplot(x='dataset', y=col, data=combined_df, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 히스토그램 비교
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        v2aix_df[col].hist(alpha=0.7, label='V2AIX', ax=axes[i], bins=30)
        veremi_df[col].hist(alpha=0.7, label='VeReMi', ax=axes[i], bins=30)
        axes[i].set_title(f'{col} Histogram')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def perform_pca_analysis(v2aix_df, veremi_df, numeric_cols):
    """PCA 분석"""
    print("\n=== PCA 분석 ===")
    
    if len(numeric_cols) < 2:
        print("❌ PCA 분석을 위한 충분한 피처가 없습니다.")
        return
    
    # 데이터 결합
    combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    
    # 결측값 처리
    X = combined_df[numeric_cols].fillna(0).values
    X = StandardScaler().fit_transform(X)
    
    # PCA 수행
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    combined_df['pca1'] = X_pca[:, 0]
    combined_df['pca2'] = X_pca[:, 1]
    
    # PCA 시각화
    plt.figure(figsize=(12, 8))
    
    # 샘플링 (너무 많으면 느려짐)
    sample_size = min(5000, len(combined_df))
    sample_df = combined_df.sample(sample_size)
    
    sns.scatterplot(x='pca1', y='pca2', hue='dataset', data=sample_df, alpha=0.6)
    plt.title(f'PCA Analysis - Feature Space Comparison\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.show()
    
    # 피처 중요도 (PC1, PC2에 대한 기여도)
    feature_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'PC1_Contribution': np.abs(pca.components_[0]),
        'PC2_Contribution': np.abs(pca.components_[1])
    }).sort_values('PC1_Contribution', ascending=False)
    
    print("\n📈 주요 피처의 PCA 기여도 (상위 10개):")
    print(feature_importance.head(10).round(4))
    
    return pca, feature_importance

def check_data_consistency(v2aix_df, veremi_df):
    """데이터 일관성 검사"""
    print("\n=== 데이터 일관성 검사 ===")
    
    # 값 범위 검사
    print("\n🔍 값 범위 검사:")
    for name, df in [("V2AIX", v2aix_df), ("VeReMi", veremi_df)]:
        print(f"\n{name}:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # 상위 5개만
            if col not in ['label']:
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")

def generate_quality_report(v2aix_df, veremi_df):
    """품질 보고서 생성"""
    print("\n=== 전처리 품질 보고서 ===")
    
    report = {
        'V2AIX_Rows': len(v2aix_df),
        'V2AIX_Columns': len(v2aix_df.columns),
        'VeReMi_Rows': len(veremi_df),
        'VeReMi_Columns': len(veremi_df.columns),
        'V2AIX_Missing_Total': v2aix_df.isnull().sum().sum(),
        'VeReMi_Missing_Total': veremi_df.isnull().sum().sum(),
        'V2AIX_Duplicates': v2aix_df.duplicated().sum(),
        'VeReMi_Duplicates': veremi_df.duplicated().sum()
    }
    
    if 'label' in v2aix_df.columns and 'label' in veremi_df.columns:
        report['V2AIX_Attack_Ratio'] = v2aix_df['label'].mean()
        report['VeReMi_Attack_Ratio'] = veremi_df['label'].mean()
    
    print("\n📋 품질 지표:")
    for key, value in report.items():
        if 'Ratio' in key:
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:,}")
    
    # 품질 점수 계산
    quality_score = 0
    max_score = 100
    
    # 결측값 점수 (20점)
    total_cells = report['V2AIX_Rows'] * report['V2AIX_Columns'] + report['VeReMi_Rows'] * report['VeReMi_Columns']
    missing_ratio = (report['V2AIX_Missing_Total'] + report['VeReMi_Missing_Total']) / total_cells
    quality_score += (1 - missing_ratio) * 20
    
    # 중복 데이터 점수 (20점)
    total_rows = report['V2AIX_Rows'] + report['VeReMi_Rows']
    duplicate_ratio = (report['V2AIX_Duplicates'] + report['VeReMi_Duplicates']) / total_rows
    quality_score += (1 - duplicate_ratio) * 20
    
    # 데이터 크기 점수 (30점)
    min_rows = min(report['V2AIX_Rows'], report['VeReMi_Rows'])
    if min_rows > 10000:
        quality_score += 30
    elif min_rows > 5000:
        quality_score += 20
    elif min_rows > 1000:
        quality_score += 10
    
    # 라벨 분포 점수 (30점)
    if 'V2AIX_Attack_Ratio' in report:
        attack_ratios = [report['V2AIX_Attack_Ratio'], report['VeReMi_Attack_Ratio']]
        balanced_score = 1 - abs(attack_ratios[0] - attack_ratios[1])
        quality_score += balanced_score * 30
    
    print(f"\n🎯 전처리 품질 점수: {quality_score:.1f}/100")
    
    if quality_score >= 80:
        print("✅ 우수한 전처리 품질")
    elif quality_score >= 60:
        print("⚠️ 보통 전처리 품질")
    else:
        print("❌ 개선이 필요한 전처리 품질")

def main():
    """메인 실행 함수"""
    # 데이터 로딩
    v2aix_df, veremi_df = load_and_validate_data()
    if v2aix_df is None or veremi_df is None:
        return
    
    # 데이터 품질 분석
    analyze_data_quality(v2aix_df, veremi_df)
    
    # 라벨 분포 분석
    analyze_label_distribution(v2aix_df, veremi_df)
    
    # 피처 통계 분석
    numeric_cols = analyze_feature_statistics(v2aix_df, veremi_df)
    
    # 피처 분포 시각화
    visualize_feature_distributions(v2aix_df, veremi_df, numeric_cols)
    
    # PCA 분석
    perform_pca_analysis(v2aix_df, veremi_df, numeric_cols)
    
    # 데이터 일관성 검사
    check_data_consistency(v2aix_df, veremi_df)
    
    # 품질 보고서 생성
    generate_quality_report(v2aix_df, veremi_df)
    
    print("\n=== 검증 완료 ===")
    print("전처리된 데이터의 품질을 종합적으로 검증했습니다.")

if __name__ == "__main__":
    main()
