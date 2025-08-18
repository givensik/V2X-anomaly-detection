import json
import pandas as pd
import numpy as np
from pathlib import Path

def process_veremi_json_log(file_path):
    """
    VeReMi JSON log 파일에서 type:3 메시지를 추출하고 파라미터를 처리합니다.
    """
    extracted_data = []
    
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                # type: 3 메시지만 처리
                if data.get("type") == 3:
                    params = {}
                    params['rcvTime'] = data.get('rcvTime')
                    params['sendTime'] = data.get('sendTime')
                    params['sender'] = data.get('sender')
                    params['messageID'] = data.get('messageID')
                    params['RSSI'] = data.get('RSSI')
                    
                    # pos 벡터를 스칼라 값으로 변환
                    pos = data.get('pos', [0, 0, 0])
                    params['pos_x'] = pos[0]
                    params['pos_y'] = pos[1]
                    params['pos_z'] = pos[2]
                    
                    # spd 벡터를 속도(크기)와 방향(heading)으로 변환
                    spd = data.get('spd', [0, 0, 0])
                    speed_magnitude = np.sqrt(spd[0]**2 + spd[1]**2)
                    heading_angle = np.degrees(np.arctan2(spd[1], spd[0]))
                    
                    params['speed_magnitude'] = speed_magnitude
                    params['heading_angle'] = heading_angle
                    
                    extracted_data.append(params)
    
    return pd.DataFrame(extracted_data)

# --- 실행 부분 ---
if __name__ == "__main__":
    data_dir = Path("results")
    sample_log_file = data_dir / "JSONlog-0-7-A0.json"
    
    if sample_log_file.exists():
        df_veremi = process_veremi_json_log(sample_log_file)
        
        if not df_veremi.empty:
            print(f"파일 '{sample_log_file.name}' 처리 완료. DataFrame의 첫 5행:")
            print(df_veremi.head())
            
            # 결과를 CSV 파일로 저장
            output_csv_path = 'veremi_processed_cam_data.csv'
            df_veremi.to_csv(output_csv_path, index=False)
            print(f"\nCAM 파라미터가 '{output_csv_path}' 파일로 저장되었습니다.")
        else:
            print("type: 3 메시지를 찾을 수 없거나 데이터프레임이 비어 있습니다.")
    else:
        print(f"오류: {sample_log_file} 파일이 존재하지 않습니다. 경로를 확인해주세요.")