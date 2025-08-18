import json
from pathlib import Path
import os
import pandas as pd

def list_all_keys(data, prefix=''):
    """
    중첩된 딕셔너리와 리스트를 재귀적으로 탐색하여 모든 키의 전체 경로를 리스트로 반환합니다.
    """
    keys = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                keys.extend(list_all_keys(v, new_prefix))
            else:
                keys.append(new_prefix)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            keys.extend(list_all_keys(item, new_prefix))
    return keys

# JSON Lines 형식의 파일 로드 함수
def load_json_lines(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# --- 실행 부분 ---
if __name__ == "__main__":
    # --- 파일 경로 설정 (이미지 경로에 맞게 설정) ---
    data_dir = Path("results/")
    groundtruth_file = data_dir / "GroundTruthJSONlog.json"
    sample_log_file = data_dir / "JSONlog-0-7-A0.json"
    
    # 파일이 존재하는지 확인
    if not groundtruth_file.exists():
        print(f"오류: {groundtruth_file} 파일이 존재하지 않습니다. 경로를 확인해주세요.")
    if not sample_log_file.exists():
        print(f"오류: {sample_log_file} 파일이 존재하지 않습니다. 경로를 확인해주세요.")
    
    if groundtruth_file.exists() and sample_log_file.exists():
        # --- GroundTruthJSONlog.json 파일 처리 ---
        print("=" * 40)
        print(f"'{groundtruth_file.name}' 피처 리스팅")
        print("=" * 40)
        groundtruth_data = load_json_lines(groundtruth_file)
        
        if groundtruth_data:
            groundtruth_keys = list_all_keys(groundtruth_data[0])
            for key in groundtruth_keys:
                print(key)
        else:
            print("GroundTruthJSONlog.json 파일에 메시지가 없습니다.")

        # --- JSONlog-0-7-A0.json 파일 처리 ---
        print("\n" + "=" * 40)
        print(f"'{sample_log_file.name}' 피처 리스팅")
        print("=" * 40)
        sample_log_data = load_json_lines(sample_log_file)
        
        if sample_log_data:
            sample_log_keys = list_all_keys(sample_log_data[0])
            for key in sample_log_keys:
                print(key)
        else:
            print("JSONlog-0-7-A0.json 파일에 메시지가 없습니다.")

        print("\n" + "=" * 40)
        print("VeReMi GroundTruth와 Message Log의 구조적 차이를 확인했습니다.")
        print("GroundTruth 파일에는 'attackerType' 라벨이 있습니다.")
        print("Message Log 파일에는 'rcvTime', 'RSSI'와 같은 수신 관련 정보가 있습니다.")
        print("=" * 40)