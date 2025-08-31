import json
import glob
import os

def extract_attacker_type1():
    """
    GroundTruthJSONlog.json에서 attackerType:1인 데이터의 messageID를 찾아서
    모든 JSONlog 파일에서 해당 messageID와 일치하는 데이터를 추출
    """
    ground_truth_file = "VeReMi_Data/results/GroundTruthJSONlog.json"
    jsonlog_pattern = "VeReMi_Data/results/JSONlog-*.json"
    output_file = "VeReMi_Data/results/attacker_type1_data_all.json"
    
    # 1단계: GroundTruthJSONlog.json에서 attackerType:1인 messageID 추출
    attacker_message_ids = set()
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('attackerType') == 1:
                            message_id = data.get('messageID')
                            if message_id is not None:
                                attacker_message_ids.add(message_id)
                    except json.JSONDecodeError as e:
                        print(f"GroundTruth JSON 파싱 오류 (라인 {line_num}): {e}")
                        continue
        
        print(f"attackerType:1인 messageID {len(attacker_message_ids)}개를 찾았습니다.")
        
        # 2단계: 모든 JSONlog 파일에서 해당 messageID와 일치하는 데이터 추출
        attacker_type1_data = []
        jsonlog_files = glob.glob(jsonlog_pattern)
        
        print(f"처리할 JSONlog 파일 {len(jsonlog_files)}개를 찾았습니다.")
        
        for jsonlog_file in sorted(jsonlog_files):
            print(f"처리 중: {os.path.basename(jsonlog_file)}")
            file_data_count = 0
            
            with open(jsonlog_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            message_id = data.get('messageID')
                            if message_id is not None and message_id in attacker_message_ids:
                                data['source_file'] = os.path.basename(jsonlog_file)  # 어느 파일에서 왔는지 기록
                                attacker_type1_data.append(data)
                                file_data_count += 1
                        except json.JSONDecodeError as e:
                            print(f"  JSONlog 파싱 오류 (라인 {line_num}): {e}")
                            continue
            
            print(f"  {os.path.basename(jsonlog_file)}에서 {file_data_count}개의 공격자 데이터 발견")
        
        # 3단계: 추출된 데이터를 새 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in attacker_type1_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"\n총 {len(attacker_type1_data)}개의 attackerType:1 데이터를 {output_file}에 저장했습니다.")
        
        # 파일별 통계 출력
        file_stats = {}
        for data in attacker_type1_data:
            source = data.get('source_file', 'unknown')
            file_stats[source] = file_stats.get(source, 0) + 1
        
        print("\n파일별 공격자 데이터 통계:")
        for file_name, count in sorted(file_stats.items()):
            print(f"  {file_name}: {count}개")
        
        # 첫 번째 데이터 샘플 출력
        if attacker_type1_data:
            print("\n첫 번째 데이터 샘플:")
            sample_data = attacker_type1_data[0].copy()
            # source_file 필드는 샘플에서 제거 (원본 데이터가 아니므로)
            if 'source_file' in sample_data:
                source_file = sample_data.pop('source_file')
                print(f"출처 파일: {source_file}")
            print(json.dumps(sample_data, indent=2, ensure_ascii=False))
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    extract_attacker_type1()
