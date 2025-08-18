import json
from pathlib import Path
import os

def list_all_keys(d, prefix=''):
    """
    중첩된 딕셔너리를 재귀적으로 탐색하여 모든 키의 전체 경로를 리스트로 반환합니다.
    """
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                keys.extend(list_all_keys(v, new_prefix))
            else:
                keys.append(new_prefix)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            new_prefix = f"{prefix}[{i}]"
            keys.extend(list_all_keys(item, new_prefix))
    return keys

# --- 실행 부분 ---
# 당신의 실제 파일 경로에 맞게 수정하세요.
sample_file_path = "json/Mobile/V2X-only/Aachen/scenarios/2024-01-21T18-06-39Z.json"

# # 테스트용 더미 파일 생성 (실제 파일이 없는 경우를 대비)
# dummy_data_for_test = {
#     "/v2x/cam": [
#         {
#             "recording_timestamp_nsec": 1705860419716179223,
#             "message": {
#                 "header": {
#                     "protocol_version": 2,
#                     "message_id": 2,
#                     "station_id": {
#                         "value": 1421989417
#                     }
#                 },
#                 "cam": {
#                     "generation_delta_time": {
#                         "value": 40214
#                     },
#                     "cam_parameters": {
#                         "basic_container": {
#                             "station_type": { "value": 5 },
#                             "reference_position": {
#                                 "latitude": { "value": 507201715 },
#                                 "longitude": { "value": 61272556 }
#                             }
#                         },
#                         "high_frequency_container": {
#                             "basic_vehicle_container_high_frequency": {
#                                 "speed": { "speed_value": { "value": 1849 } },
#                                 "heading": { "heading_value": { "value": 3166 } }
#                             }
#                         }
#                     }
#                 }
#             }
#         }
#     ]
# }

# # 더미 파일 생성
# os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
# with open(sample_file_path, 'w') as f:
#     json.dump(dummy_data_for_test, f, indent=2)


with open(sample_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# CAM 메시지 리스트의 첫 번째 메시지 딕셔너리 가져오기
first_cam_message = data.get('/v2x/cam', [])[0] if data.get('/v2x/cam') else {}

# 모든 키의 경로를 출력
print("--- 모든 CAM 메시지 파라미터 경로 ---")
all_keys = list_all_keys(first_cam_message)
for key in all_keys:
    print(key)