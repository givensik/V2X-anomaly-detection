import json
import pandas as pd
import os

def extract_cam_params(cam_data):
    """
    CAM 메시지 딕셔너리에서 핵심 파라미터를 추출하여 딕셔너리로 반환합니다.
    누락된 필드는 None으로 처리하여 오류를 방지합니다.
    """
    params = {}
    
    # 상위 계층 데이터 추출 (recording_timestamp_nsec)
    params['recording_timestamp_nsec'] = cam_data.get('recording_timestamp_nsec')

    # 'message' 컨테이너 접근
    message = cam_data.get('message', {})
    header = message.get('header', {})
    cam = message.get('cam', {})
    
    # 헤더 정보
    params['station_id'] = header.get('station_id', {}).get('value')
    
    # CAM 파라미터 컨테이너
    cam_params = cam.get('cam_parameters', {})
    
    # basic_container에서 위치 정보 추출 및 스케일링
    basic_container = cam_params.get('basic_container', {})
    ref_pos = basic_container.get('reference_position', {})
    
    lat_value = ref_pos.get('latitude', {}).get('value')
    params['latitude'] = lat_value * 1e-7 if lat_value is not None else None
    
    lon_value = ref_pos.get('longitude', {}).get('value')
    params['longitude'] = lon_value * 1e-7 if lon_value is not None else None
    
    alt_value = ref_pos.get('altitude', {}).get('altitude_value', {}).get('value')
    params['altitude'] = alt_value
    
    # high_frequency_container에서 동적 정보 추출 및 스케일링
    high_freq_container = cam_params.get('high_frequency_container', {}).get('basic_vehicle_container_high_frequency', {})
    
    speed_value = high_freq_container.get('speed', {}).get('speed_value', {}).get('value')
    params['speed'] = speed_value * 0.02 if speed_value is not None else None
    
    heading_value = high_freq_container.get('heading', {}).get('heading_value', {}).get('value')
    params['heading'] = heading_value * 0.1 if heading_value is not None else None

    params['longitudinal_acceleration'] = high_freq_container.get('longitudinal_acceleration', {}).get('longitudinal_acceleration_value', {}).get('value')
    params['yaw_rate'] = high_freq_container.get('yaw_rate', {}).get('yaw_rate_value', {}).get('value')
    
    # low_frequency_container에서 path_history 추출 (리스트 평탄화)
    low_freq_container = cam_params.get('low_frequency_container', {}).get('basic_vehicle_container_low_frequency', {})
    path_history = low_freq_container.get('path_history', {}).get('array', [])
    
    # path_history의 각 포인트를 개별 열로 추출
    for i, path_point in enumerate(path_history):
        path_pos = path_point.get('path_position', {})
        params[f'path_{i}_delta_lat'] = path_pos.get('delta_latitude', {}).get('value')
        params[f'path_{i}_delta_lon'] = path_pos.get('delta_longitude', {}).get('value')
        params[f'path_{i}_delta_time'] = path_point.get('path_delta_time', {}).get('value')
    
    return params

def process_v2aix_json_from_file(file_path):
    """
    V2AIX JSON 파일에서 CAM 메시지 배열을 추출하여 DataFrame으로 만듭니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

    # "/v2x/cam" 키를 사용하여 CAM 메시지 배열을 추출
    cam_messages = json_data.get('/v2x/cam', [])

    if not cam_messages:
        print(f"파일 '{file_path}'에 '/v2x/cam' 키가 없거나 비어 있습니다.")
        return None

    extracted_data = [extract_cam_params(msg) for msg in cam_messages]
    
    df = pd.DataFrame(extracted_data)
    return df

if __name__ == "__main__":
    # --- 파일 경로 설정 ---
    # 당신의 실제 파일 경로로 수정하세요.
    file_path_to_process = 'V2AIX_Data/json/Mobile/V2X-only/Aachen/scenarios/joined.json'
    
    # # --- 더미 파일 생성 (테스트용) ---
    # # 실제 파일이 없는 경우를 대비하여 여러 개의 CAM 메시지가 담긴 더미 파일을 생성합니다.
    # # 이 부분은 실제 데이터를 처리할 때는 주석 처리하거나 삭제하세요.
    # dummy_data = {
    #     "/v2x/cam": [
    #         {
    #             "recording_timestamp_nsec": 1705860419716179223,
    #             "message": {
    #                 "header": { "station_id": { "value": 1421989417 } },
    #                 "cam": {
    #                     "generation_delta_time": { "value": 40214 },
    #                     "cam_parameters": {
    #                         "basic_container": {
    #                             "reference_position": { "latitude": { "value": 507201715 }, "longitude": { "value": 61272556 } },
    #                         },
    #                         "high_frequency_container": {
    #                             "basic_vehicle_container_high_frequency": {
    #                                 "speed": { "speed_value": { "value": 1849 } },
    #                                 "heading": { "heading_value": { "value": 3166 } },
    #                             }
    #                         },
    #                         "low_frequency_container": {
    #                             "basic_vehicle_container_low_frequency": {
    #                                 "path_history": { "array": [] }
    #                             }
    #                         }
    #                     }
    #                 }
    #             }
    #         },
    #         {
    #             "recording_timestamp_nsec": 1705860420716179223,
    #             "message": {
    #                 "header": { "station_id": { "value": 1421989418 } },
    #                 "cam": {
    #                     "generation_delta_time": { "value": 41214 },
    #                     "cam_parameters": {
    #                         "basic_container": {
    #                             "reference_position": { "latitude": { "value": 507210000 }, "longitude": { "value": 61280000 } },
    #                         },
    #                         "high_frequency_container": {
    #                             "basic_vehicle_container_high_frequency": {
    #                                 "speed": { "speed_value": { "value": 1900 } },
    #                                 "heading": { "heading_value": { "value": 3100 } },
    #                             }
    #                         },
    #                         "low_frequency_container": {
    #                             "basic_vehicle_container_low_frequency": {
    #                                 "path_history": { "array": [
    #                                     {"path_position": {"delta_latitude": {"value": -100}, "delta_longitude": {"value": 200}, "path_delta_time": {"value": 100}}}
    #                                 ]}
    #                             }
    #                         }
    #                     }
    #                 }
    #             }
    #         }
    #     ]
    # }
    
    # # 더미 파일을 생성
    # os.makedirs(os.path.dirname(file_path_to_process), exist_ok=True)
    # with open(file_path_to_process, "w", encoding='utf-8') as f:
    #     json.dump(dummy_data, f, indent=2)

    # --- 메인 로직 실행 ---
    df_v2aix = process_v2aix_json_from_file(file_path_to_process)
    
    if df_v2aix is not None:
        print(f"파일 '{file_path_to_process}' 처리 완료. DataFrame의 첫 5행:")
        print(df_v2aix.head())
        
        # 결과를 CSV 파일로 저장
        output_csv_path = 'v2aix_processed_cam_data.csv'
        df_v2aix.to_csv(output_csv_path, index=False)
        print(f"\nCAM 파라미터가 '{output_csv_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    # 1. 테스트할 파일의 경로를 지정
    file_path_to_process = 'json/Mobile/V2X-only/Aachen/scenarios/2024-01-21T18-06-39Z.json'
    
    # 2. 위에서 만든 함수를 호출하여 파일을 처리
    df_v2aix = process_v2aix_json_from_file(file_path_to_process)
    
    # 3. 결과 DataFrame을 출력하고 파일로 저장
    if df_v2aix is not None:
        print(df_v2aix.head())
        df_v2aix.to_csv('processed_2024-01-21_cam_data.csv', index=False)