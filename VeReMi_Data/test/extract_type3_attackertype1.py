import json
import glob
import os
import re
from collections import defaultdict

def extract_type3_from_attackertype1():
    """
    VeReMi_Data/all/ 안에서 AttackerType1 시나리오들을 찾아서
    JSONlog*.json 파일에서 type3 데이터만 추출하여 저장
    """
    base_dir = "VeReMi_Data/all"
    output_dir = "VeReMi_Data/test/type3"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1단계: VeReMi_Data/all/ 하위 디렉토리들 찾기
        scenario_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # results 디렉토리가 있는지 확인
                results_path = os.path.join(item_path, "veins-maat", "simulations", "securecomm2018", "results")
                if os.path.exists(results_path):
                    scenario_dirs.append((item, results_path))
        
        print(f"총 {len(scenario_dirs)}개의 시나리오 디렉토리를 찾았습니다.")
        
        all_attackers_summary = []
        processed_count = 0
        
        # 2단계: AttackerType1 시나리오들만 처리
        attacker_type1_scenarios = []
        for scenario_name, results_path in scenario_dirs:
            # AttackerType*.sca 파일에서 공격자 유형 파악
            sca_files = glob.glob(os.path.join(results_path, "AttackerType*.sca"))
            if sca_files:
                sca_file = sca_files[0]
                sca_filename = os.path.basename(sca_file)
                match = re.match(r'AttackerType(\d+)', sca_filename)
                if match and int(match.group(1)) == 1:  # AttackerType1만 선택
                    attacker_type1_scenarios.append((scenario_name, results_path))
        
        print(f"AttackerType1 시나리오 {len(attacker_type1_scenarios)}개를 찾았습니다.")
        
        # 3단계: AttackerType1 시나리오들에서 type3 데이터 추출
        for scenario_name, results_path in attacker_type1_scenarios:
            processed_count += 1
            print(f"\n[{processed_count}/{len(attacker_type1_scenarios)}] 처리 중: {scenario_name}")
            
            # JSONlog*.json 파일들에서 type3 데이터 추출
            jsonlog_pattern = os.path.join(results_path, "JSONlog-*.json")
            jsonlog_files = glob.glob(jsonlog_pattern)
            
            type3_data = []
            
            for jsonlog_file in sorted(jsonlog_files):
                file_type3_count = 0
                with open(jsonlog_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                # type3 데이터인지 확인 (여러 가능한 필드명 확인)
                                message_type = data.get('type') or data.get('messageType') or data.get('msgType')
                                if message_type == 3:
                                    data['scenario'] = scenario_name
                                    data['source_file'] = os.path.basename(jsonlog_file)
                                    type3_data.append(data)
                                    file_type3_count += 1
                            except json.JSONDecodeError as e:
                                continue
                
                if file_type3_count > 0:
                    print(f"  {os.path.basename(jsonlog_file)}: {file_type3_count}개의 type3 데이터 발견")
            
            print(f"  총 {len(type3_data)}개의 type3 데이터 발견")
            
            # 시나리오별 type3 데이터 저장
            if type3_data:
                scenario_output_file = os.path.join(output_dir, f"type3_{scenario_name}.json")
                with open(scenario_output_file, 'w', encoding='utf-8') as f:
                    for data in type3_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                print(f"  저장됨: {scenario_output_file}")
                
                # 요약 정보 수집
                summary_info = {
                    "scenario": scenario_name,
                    "attacker_type": 1,
                    "type3_messages_count": len(type3_data),
                    "jsonlog_files_count": len(jsonlog_files)
                }
                all_attackers_summary.append(summary_info)
            else:
                print(f"  {scenario_name}에서 type3 데이터를 찾지 못했습니다.")
        
        # 4단계: 전체 통계 출력 및 요약 파일 생성
        print("\n" + "="*80)
        print("AttackerType1 시나리오별 type3 데이터 요약")
        print("="*80)
        
        total_scenarios = len(all_attackers_summary)
        total_type3_messages = 0
        
        for summary in all_attackers_summary:
            total_type3_messages += summary['type3_messages_count']
            
            print(f"\n시나리오: {summary['scenario']}")
            print(f"  공격자 유형: AttackerType{summary['attacker_type']}")
            print(f"  type3 메시지 수: {summary['type3_messages_count']}개")
            print(f"  JSONlog 파일 수: {summary['jsonlog_files_count']}개")
        
        print(f"\n" + "="*80)
        print("전체 통계")
        print("="*80)
        print(f"처리된 AttackerType1 시나리오 수: {total_scenarios}개")
        print(f"총 type3 메시지 수: {total_type3_messages}개")
        
        # 전체 요약 정보를 JSON 파일로 저장
        summary_data = {
            "attacker_type": 1,
            "total_scenarios": total_scenarios,
            "total_type3_messages": total_type3_messages,
            "scenarios": all_attackers_summary
        }
        
        summary_file = os.path.join(output_dir, "type3_attackertype1_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n전체 요약 정보가 {summary_file}에 저장되었습니다.")
        print(f"각 시나리오별 type3 데이터가 {output_dir} 디렉토리에 저장되었습니다.")
        
        # 모든 type3 데이터를 하나의 파일로 합치기
        all_type3_file = os.path.join(output_dir, "all_type3_attackertype1.json")
        all_type3_count = 0
        
        with open(all_type3_file, 'w', encoding='utf-8') as outf:
            for summary in all_attackers_summary:
                scenario_file = os.path.join(output_dir, f"type3_{summary['scenario']}.json")
                if os.path.exists(scenario_file):
                    with open(scenario_file, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            outf.write(line)
                            all_type3_count += 1
        
        print(f"모든 type3 데이터 {all_type3_count}개가 {all_type3_file}에 통합 저장되었습니다.")
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    extract_type3_from_attackertype1()
