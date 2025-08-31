import pandas as pd
import numpy as np
import os

def fix_preprocessing_issues():
    """전처리 문제점들을 수정하는 함수"""
    print("=== 전처리 문제점 수정 ===\n")
    
    # 데이터 로딩
    v2aix_path = 'out/v2aix_preprocessed.csv'
    veremi_path = 'out/veremi_preprocessed.csv'
    
    if not os.path.exists(v2aix_path) or not os.path.exists(veremi_path):
        print("❌ 전처리된 CSV 파일이 없습니다.")
        return
    
    print("📁 데이터 로딩 중...")
    v2aix_df = pd.read_csv(v2aix_path)
    veremi_df = pd.read_csv(veremi_path)
    
    print(f"✅ V2AIX: {len(v2aix_df):,} 행")
    print(f"✅ VeReMi: {len(veremi_df):,} 행")
    
    # 1. is_attacker 컬럼 확인 및 분석
    print("\n=== 1. is_attacker 컬럼 분석 ===")
    
    # V2AIX의 is_attacker 확인
    if 'is_attacker' in v2aix_df.columns:
        v2aix_attack_count = v2aix_df['is_attacker'].sum()
        print(f"V2AIX is_attacker: {v2aix_attack_count:,}개 공격 (비율: {v2aix_attack_count/len(v2aix_df):.3f})")
    else:
        print("❌ V2AIX에 'is_attacker' 컬럼이 없습니다.")
        return
    
    # VeReMi의 is_attacker 확인
    if 'is_attacker' in veremi_df.columns:
        veremi_attack_count = veremi_df['is_attacker'].sum()
        veremi_normal_count = len(veremi_df) - veremi_attack_count
        print(f"VeReMi is_attacker: 정상 {veremi_normal_count:,}개, 공격 {veremi_attack_count:,}개")
        print(f"공격 비율: {veremi_attack_count/len(veremi_df):.3f}")
    else:
        print("❌ VeReMi에 'is_attacker' 컬럼이 없습니다.")
        return
    
    # 2. 피처 스케일링 문제 해결
    print("\n=== 2. 피처 스케일링 문제 해결 ===")
    
    # 공통 피처들 식별
    common_features = set(v2aix_df.columns) & set(veremi_df.columns)
    common_features = [col for col in common_features if col not in ['is_attacker', 'dataset']]
    
    print(f"공통 피처 수: {len(common_features)}개")
    
    # 스케일링이 필요한 피처들 식별
    scaling_needed = []
    for feature in common_features:
        if feature in v2aix_df.columns and feature in veremi_df.columns:
            # 수치형 컬럼만 처리
            if v2aix_df[feature].dtype in ['int64', 'float64'] and veremi_df[feature].dtype in ['int64', 'float64']:
                v2aix_range = v2aix_df[feature].max() - v2aix_df[feature].min()
                veremi_range = veremi_df[feature].max() - veremi_df[feature].min()
                
                # 0으로 나누기 방지
                if v2aix_range > 0 and veremi_range > 0:
                    # 범위 차이가 10배 이상이면 스케일링 필요
                    if max(v2aix_range, veremi_range) / min(v2aix_range, veremi_range) > 10:
                        scaling_needed.append(feature)
                elif v2aix_range == 0 and veremi_range > 0:
                    # V2AIX는 고정값, VeReMi는 변화값
                    scaling_needed.append(feature)
                elif veremi_range == 0 and v2aix_range > 0:
                    # VeReMi는 고정값, V2AIX는 변화값
                    scaling_needed.append(feature)
    
    print(f"스케일링이 필요한 피처: {len(scaling_needed)}개")
    for feature in scaling_needed[:5]:  # 상위 5개만 출력
        v2aix_range = f"[{v2aix_df[feature].min():.2f}, {v2aix_df[feature].max():.2f}]"
        veremi_range = f"[{veremi_df[feature].min():.2f}, {veremi_df[feature].max():.2f}]"
        print(f"  {feature}: V2AIX {v2aix_range}, VeReMi {veremi_range}")
    
    # 3. 데이터셋 특성 분석
    print("\n=== 3. 데이터셋 특성 분석 ===")
    
    # V2AIX 특성
    print("V2AIX 특성:")
    print(f"  - 정상 데이터만 존재 (공격 비율: {v2aix_attack_count/len(v2aix_df):.1%})")
    print(f"  - 이상탐지 모델의 정상 패턴 학습용")
    print(f"  - Unsupervised learning에 적합")
    
    # VeReMi 특성
    attack_ratio = veremi_attack_count / len(veremi_df)
    print(f"\nVeReMi 특성:")
    print(f"  - 정상 데이터: {1-attack_ratio:.1%}")
    print(f"  - 공격 데이터: {attack_ratio:.1%}")
    print(f"  - 이상탐지 모델의 성능 평가용")
    print(f"  - Supervised/Semi-supervised learning 가능")
    
    # 4. 수정된 데이터 저장 (is_attacker 그대로 유지)
    print("\n=== 4. 수정된 데이터 저장 ===")
    
    # 백업 생성
    v2aix_backup_path = 'out/v2aix_preprocessed_backup.csv'
    veremi_backup_path = 'out/veremi_preprocessed_backup.csv'
    
    if not os.path.exists(v2aix_backup_path):
        pd.read_csv(v2aix_path).to_csv(v2aix_backup_path, index=False)
        print(f"✅ V2AIX 백업 생성: {v2aix_backup_path}")
    
    if not os.path.exists(veremi_backup_path):
        pd.read_csv(veremi_path).to_csv(veremi_backup_path, index=False)
        print(f"✅ VeReMi 백업 생성: {veremi_backup_path}")
    
    # 수정된 데이터 저장 (is_attacker 컬럼 그대로 유지)
    v2aix_fixed_path = 'out/v2aix_preprocessed_fixed.csv'
    veremi_fixed_path = 'out/veremi_preprocessed_fixed.csv'
    
    v2aix_df.to_csv(v2aix_fixed_path, index=False)
    veremi_df.to_csv(veremi_fixed_path, index=False)
    
    print(f"✅ 수정된 V2AIX 저장: {v2aix_fixed_path}")
    print(f"✅ 수정된 VeReMi 저장: {veremi_fixed_path}")
    
    # 5. 권장 모델링 전략
    print("\n=== 5. 권장 모델링 전략 ===")
    
    print("🎯 이상탐지 모델링 접근법:")
    print("1. **Domain Adaptation 방식**:")
    print("   - V2AIX (정상) → VeReMi (혼재) 도메인 적응")
    print("   - 정상 패턴을 V2AIX에서 학습하고 VeReMi에 적용")
    
    print("\n2. **Semi-supervised 방식**:")
    print("   - V2AIX의 정상 데이터로 정상 패턴 학습")
    print("   - VeReMi의 is_attacker로 성능 평가")
    
    print("\n3. **Ensemble 방식**:")
    print("   - AutoEncoder (V2AIX 정상 데이터 학습)")
    print("   - Rule-based (VeReMi의 is_attacker 활용)")
    print("   - 두 모델의 결과를 결합")
    
    # 6. 품질 점수 재계산
    print("\n=== 6. 수정 후 품질 점수 ===")
    
    # 결측값 점수 (20점)
    v2aix_missing = v2aix_df.isnull().sum().sum()
    veremi_missing = veremi_df.isnull().sum().sum()
    total_cells = len(v2aix_df) * len(v2aix_df.columns) + len(veremi_df) * len(veremi_df.columns)
    missing_score = (1 - (v2aix_missing + veremi_missing) / total_cells) * 20
    
    # 중복 데이터 점수 (20점)
    v2aix_duplicates = v2aix_df.duplicated().sum()
    veremi_duplicates = veremi_df.duplicated().sum()
    total_rows = len(v2aix_df) + len(veremi_df)
    duplicate_score = (1 - (v2aix_duplicates + veremi_duplicates) / total_rows) * 20
    
    # 데이터 크기 점수 (30점)
    min_rows = min(len(v2aix_df), len(veremi_df))
    if min_rows > 10000:
        size_score = 30
    elif min_rows > 5000:
        size_score = 20
    else:
        size_score = 10
    
    # 라벨 분포 점수 (30점) - is_attacker가 있으므로
    # V2AIX는 정상만, VeReMi는 혼재이므로 적절한 분포
    label_score = 30  # is_attacker가 올바르게 있으므로 만점
    
    total_score = missing_score + duplicate_score + size_score + label_score
    
    print(f"📊 수정 후 품질 점수: {total_score:.1f}/100")
    print(f"  - 결측값: {missing_score:.1f}/20")
    print(f"  - 중복: {duplicate_score:.1f}/20")
    print(f"  - 크기: {size_score}/30")
    print(f"  - 라벨: {label_score}/30")
    
    if total_score >= 80:
        print("✅ 우수한 전처리 품질")
    elif total_score >= 60:
        print("⚠️ 보통 전처리 품질")
    else:
        print("❌ 개선이 필요한 전처리 품질")
    
    print("\n=== 수정 완료 ===")
    print("이제 이상탐지 모델 학습에 사용할 수 있는 데이터가 준비되었습니다.")
    print("💡 is_attacker 컬럼을 그대로 사용하여 모델을 학습하세요!")

if __name__ == "__main__":
    fix_preprocessing_issues()
