import json
from pathlib import Path
import os

def list_all_keys(data, prefix=''):
    print("data ", data)
    """
    중첩된 딕셔너리와 리스트를 재귀적으로 탐색하여 모든 키의 전체 경로를 리스트로 반환합니다.
    """
    keys = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                print("v ", v)
                keys.extend(list_all_keys(v, new_prefix))
            else:
                keys.append(new_prefix)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            print("new_prefix ", new_prefix)
            keys.extend(list_all_keys(item, new_prefix))
    return keys

def load_json_lines_with_filter(file_path, message_type=None):
    """
    JSON Lines 형식의 파일을 읽고, 특정 메시지 타입만 필터링합니다.
    """
    filtered_data = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if message_type is None or data.get("type") == message_type:
                        filtered_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSON 디코딩 오류: {e} (라인: '{line.strip()}')")
    return filtered_data

# --- 실행 부분 ---
if __name__ == "__main__":
    # --- 파일 경로 설정 ---
    # 실제 파일 경로에 맞게 설정하세요.
    data_dir = Path("results/")
    sample_log_file = data_dir / "JSONlog-0-7-A0.json"
    
    if not sample_log_file.exists():
        print(f"오류: {sample_log_file} 파일이 존재하지 않습니다. 경로를 확인해주세요.")
    else:
        # 파일에서 type: 3 메시지만 로드
        type3_messages = load_json_lines_with_filter(sample_log_file, message_type=3)
        
        if type3_messages:
            # 첫 번째 'type: 3' 메시지의 모든 키를 리스팅
            type3_keys = list_all_keys(type3_messages[0])
            print("--- VeReMi 'type: 3' 메시지의 모든 키 경로 ---")
            for key in type3_keys:
                print(key)
        else:
            print("파일에 'type: 3' 메시지가 없습니다.")