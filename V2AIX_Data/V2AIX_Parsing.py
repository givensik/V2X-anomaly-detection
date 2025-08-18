import json
import pandas as pd
from pathlib import Path

# ====== 사용자 경로 설정 ======
# 예시: "Mobile/V2X-only/Aachen/scenarios"
SCENARIO_DIR = Path("json/Mobile/V2X-with-Sensor-Context/Aachen/scenarios")  

# ====== CAM 파싱 함수 ======
def parse_cam_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "/v2x/cam" not in data:
        return pd.DataFrame()  # CAM이 없는 경우 빈 DF 반환

    records = []
    for entry in data["/v2x/cam"]:
        ts_nsec = entry.get("recording_timestamp_nsec")
        cam_msg = entry["message"]["cam"]

        # GNSS 위치/속도 정보 추출
        try:
            ref_pos = cam_msg["basic_container"]["reference_position"]
            lat = ref_pos.get("latitude")
            lon = ref_pos.get("longitude")
        except KeyError:
            lat, lon = None, None

        try:
            speed_val = cam_msg["high_frequency_container"]["speed"]["speed_value"]
        except KeyError:
            speed_val = None

        station_id = entry["message"]["header"]["station_id"]["value"]

        records.append({
            "timestamp_nsec": ts_nsec,
            "timestamp_sec": ts_nsec / 1e9 if ts_nsec else None,
            "station_id": station_id,
            "latitude": lat,
            "longitude": lon,
            "speed_kph": speed_val * 3.6 if speed_val is not None else None
        })
    
    return pd.DataFrame(records)

# ====== 모든 scenario/*.json 처리 ======
all_cam_df = pd.DataFrame()
for json_file in SCENARIO_DIR.glob("*.json"):
    print(f"Processing {json_file.name}...")
    df = parse_cam_from_json(json_file)
    all_cam_df = pd.concat([all_cam_df, df], ignore_index=True)

# ====== CSV 저장 ======
all_cam_df.to_csv("V2AIX_CAM_scenarios.csv", index=False)

print(f"총 {len(all_cam_df)}개의 CAM 레코드 추출 완료")
print(all_cam_df.head())

