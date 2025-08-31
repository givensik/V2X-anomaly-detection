#!/usr/bin/env python3
"""
VeReMi 데이터셋에서 Type 2 공격이 있는 파일을 찾는 스크립트
"""

import json
from pathlib import Path
from collections import defaultdict

def find_files_with_attack_types():
    """각 공격 타입이 있는 파일들을 찾기"""
    
    veremi_root = Path('VeReMi_Data/all')
    gt_files = list(veremi_root.rglob('**/GroundTruthJSONlog.json'))
    
    print(f'Scanning {len(gt_files)} GroundTruth files for attack types...')
    
    files_by_type = defaultdict(list)
    
    for i, gt_file in enumerate(gt_files):
        if i % 25 == 0:
            print(f'Processed {i}/{len(gt_files)} files...')
        
        try:
            attack_types_in_file = set()
            with open(gt_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num > 200:  # 파일당 최대 200줄만 확인
                        break
                    try:
                        entry = json.loads(line.strip())
                        att_type = entry.get('attackerType', 0)
                        if att_type > 0:
                            attack_types_in_file.add(att_type)
                    except:
                        continue
            
            # 이 파일에서 발견된 공격 타입들을 기록
            for att_type in attack_types_in_file:
                files_by_type[att_type].append(str(gt_file))
                
        except Exception as e:
            continue
    
    print('\nFiles by attack type:')
    for att_type in sorted(files_by_type.keys()):
        files = files_by_type[att_type]
        print(f'\nAttack Type {att_type}: {len(files)} files')
        # 처음 3개 파일만 출력
        for i, file_path in enumerate(files[:3]):
            file_name = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
            dir_name = file_path.split('\\')[-4:-1] if '\\' in file_path else file_path.split('/')[-4:-1]
            print(f'  {i+1}. {"/".join(dir_name[-2:])} -> {file_name}')
        
        if len(files) > 3:
            print(f'  ... and {len(files)-3} more files')
    
    return files_by_type

if __name__ == "__main__":
    files_by_type = find_files_with_attack_types()
    
    # Type 2 공격이 있는 파일이 있다면 하나 테스트
    if 2 in files_by_type:
        print(f'\n=== Testing Type 2 Attack File ===')
        type2_file = files_by_type[2][0]
        print(f'Testing file: {type2_file}')
        
        # 해당 파일의 JSON 로그 찾기
        gt_path = Path(type2_file)
        json_logs = list(gt_path.parent.glob('JSONlog-*.json'))
        json_logs = [f for f in json_logs if f.name != 'GroundTruthJSONlog.json']
        
        if json_logs:
            print(f'Found {len(json_logs)} JSON log files in the same directory')
            
            # 첫 번째 로그 파일로 테스트
            log_path = str(json_logs[0])
            print(f'Testing with: {json_logs[0].name}')
            
            # 실제 로딩 테스트
            import sys
            sys.path.append('.')
            from v2x_preprocessing_lstm import V2XDataPreprocessor
            
            preprocessor = V2XDataPreprocessor()
            
            print('\n1. Loading all types:')
            df_all = preprocessor.load_veremi_data(log_path, type2_file, target_attacker_types=None)
            if not df_all.empty:
                stats = df_all['attacker_type'].value_counts().sort_index()
                for att_type, count in stats.items():
                    att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
                    print(f'  {att_name}: {count} records')
            
            print('\n2. Loading only Type 2 + Normal:')
            df_type2 = preprocessor.load_veremi_data(log_path, type2_file, target_attacker_types=[0, 2])
            if not df_type2.empty:
                stats = df_type2['attacker_type'].value_counts().sort_index()
                for att_type, count in stats.items():
                    att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
                    print(f'  {att_name}: {count} records')
    else:
        print('\nNo Type 2 attack files found in the scanned portion.')
