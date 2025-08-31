#!/usr/bin/env python3
"""
VeReMi 데이터셋의 공격 타입 선택 기능 테스트 스크립트
"""

import sys
sys.path.append('.')
from v2x_preprocessing_lstm import V2XDataPreprocessor, iter_veremi_pairs
import pandas as pd

def test_attacker_type_filtering():
    """공격 타입 필터링 기능 테스트"""
    
    preprocessor = V2XDataPreprocessor()
    veremi_root = "VeReMi_Data/all"
    
    # 몇 개 파일만 테스트
    all_pairs = list(iter_veremi_pairs(veremi_root))
    test_pairs = all_pairs[:5]  # 처음 5개 파일만
    
    print("=== Testing Attacker Type Filtering ===\n")
    
    # 테스트 1: 모든 타입 포함 (기본값)
    print("1. Loading all attacker types (default):")
    all_dfs = []
    for i, (log_path, gt_path) in enumerate(test_pairs):
        print(f"  Processing file {i+1}: {log_path.split('/')[-1] if '/' in log_path else log_path.split('\\')[-1]}")
        df = preprocessor.load_veremi_data(log_path, gt_path, target_attacker_types=None)
        if not df.empty:
            all_dfs.append(df)
    
    if all_dfs:
        combined_all = pd.concat(all_dfs, ignore_index=True)
        print(f"  Total records: {len(combined_all)}")
        attacker_stats = combined_all['attacker_type'].value_counts().sort_index()
        for att_type, count in attacker_stats.items():
            att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
            print(f"    {att_name}: {count} records")
    
    print("\n" + "="*50 + "\n")
    
    # 테스트 2: Type 2 공격만 포함
    print("2. Loading only Type 2 attacks + Normal:")
    type2_dfs = []
    for i, (log_path, gt_path) in enumerate(test_pairs):
        print(f"  Processing file {i+1}: {log_path.split('/')[-1] if '/' in log_path else log_path.split('\\')[-1]}")
        df = preprocessor.load_veremi_data(log_path, gt_path, target_attacker_types=[0, 2])
        if not df.empty:
            type2_dfs.append(df)
    
    if type2_dfs:
        combined_type2 = pd.concat(type2_dfs, ignore_index=True)
        print(f"  Total records: {len(combined_type2)}")
        attacker_stats = combined_type2['attacker_type'].value_counts().sort_index()
        for att_type, count in attacker_stats.items():
            att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
            print(f"    {att_name}: {count} records")
    
    print("\n" + "="*50 + "\n")
    
    # 테스트 3: 여러 공격 타입 선택
    print("3. Loading Type 1, 2, 4 attacks + Normal:")
    multi_dfs = []
    for i, (log_path, gt_path) in enumerate(test_pairs):
        print(f"  Processing file {i+1}: {log_path.split('/')[-1] if '/' in log_path else log_path.split('\\')[-1]}")
        df = preprocessor.load_veremi_data(log_path, gt_path, target_attacker_types=[0, 1, 2, 4])
        if not df.empty:
            multi_dfs.append(df)
    
    if multi_dfs:
        combined_multi = pd.concat(multi_dfs, ignore_index=True)
        print(f"  Total records: {len(combined_multi)}")
        attacker_stats = combined_multi['attacker_type'].value_counts().sort_index()
        for att_type, count in attacker_stats.items():
            att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
            print(f"    {att_name}: {count} records")
    
    print("\n=== Test Completed Successfully! ===")

if __name__ == "__main__":
    test_attacker_type_filtering()
