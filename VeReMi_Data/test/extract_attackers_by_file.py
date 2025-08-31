import json
import glob
import os
import re
from collections import defaultdict

def extract_attackers_by_directory():
    """
    VeReMi_Data/all/ 안의 각 디렉토리별로 공격자를 찾아서 정리
    각 디렉토리의 results 안에 있는 AttackerType*.sca 파일로 공격자 유형을 파악하고
    GroundTruthJSONlog.json과 JSONlog*.json을 분석하여 공격자 sender를 추출
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
        
        # 2단계: 각 시나리오 디렉토리 처리
        for scenario_name, results_path in scenario_dirs:
            processed_count += 1
            print(f"\n[{processed_count}/{len(scenario_dirs)}] 처리 중: {scenario_name}")
            
            # AttackerType*.sca 파일에서 공격자 유형 파악
            sca_files = glob.glob(os.path.join(results_path, "AttackerType*.sca"))
            attacker_type = None
            if sca_files:
                sca_file = sca_files[0]  # 첫 번째 sca 파일 사용
                sca_filename = os.path.basename(sca_file)
                # AttackerType1-start=3,0.1-#0.sca -> AttackerType1 추출
                match = re.match(r'AttackerType(\d+)', sca_filename)
                if match:
                    attacker_type = int(match.group(1))
                print(f"  공격자 유형: AttackerType{attacker_type} (from {sca_filename})")
            else:
                print(f"  경고: AttackerType*.sca 파일을 찾을 수 없습니다.")
                continue
            
            # GroundTruthJSONlog.json에서 공격자 messageID 추출
            ground_truth_file = os.path.join(results_path, "GroundTruthJSONlog.json")
            if not os.path.exists(ground_truth_file):
                print(f"  경고: GroundTruthJSONlog.json 파일이 없습니다.")
                continue
                
            attacker_message_ids = set()
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get('attackerType', 0) > 0:  # 공격자인 경우
                                message_id = data.get('messageID')
                                if message_id is not None:
                                    attacker_message_ids.add(message_id)
                        except json.JSONDecodeError as e:
                            print(f"    GroundTruth JSON 파싱 오류 (라인 {line_num}): {e}")
                            continue
            
            print(f"  공격자 messageID {len(attacker_message_ids)}개 발견")
            
            # JSONlog*.json 파일들에서 공격자 데이터 추출
            jsonlog_pattern = os.path.join(results_path, "JSONlog-*.json")
            jsonlog_files = glob.glob(jsonlog_pattern)
            
            scenario_attacker_data = []
            attacker_senders = set()  # 공격자 sender들을 추적
            
            for jsonlog_file in sorted(jsonlog_files):
                with open(jsonlog_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                message_id = data.get('messageID')
                                if message_id is not None and message_id in attacker_message_ids:
                                    sender = data.get('senderID') or data.get('sender') or data.get('nodeID')
                                    if sender:
                                        attacker_senders.add(sender)
                                    data['attackerType'] = attacker_type
                                    data['scenario'] = scenario_name
                                    scenario_attacker_data.append(data)
                            except json.JSONDecodeError as e:
                                continue
            
            print(f"  공격자 데이터 {len(scenario_attacker_data)}개, 공격자 sender {len(attacker_senders)}개 발견")
            print(f"  공격자 sender들: {sorted(list(attacker_senders))}")
            
            # 시나리오별 결과 저장
            if scenario_attacker_data:
                scenario_output_file = os.path.join(output_dir, f"attackers_{scenario_name}.json")
                with open(scenario_output_file, 'w', encoding='utf-8') as f:
                    for data in scenario_attacker_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                print(f"  저장됨: {scenario_output_file}")
            
            # 요약 정보 수집
            summary_info = {
                "scenario": scenario_name,
                "attacker_type": attacker_type,
                "attacker_senders": sorted(list(attacker_senders)),
                "total_attacker_messages": len(scenario_attacker_data),
                "jsonlog_files_count": len(jsonlog_files)
            }
            all_attackers_summary.append(summary_info)
        
        # 3단계: 전체 통계 출력 및 요약 파일 생성
        print("\n" + "="*80)
        print("전체 시나리오별 공격자 요약")
        print("="*80)
        
        # 공격자 유형별 통계
        attacker_type_stats = defaultdict(int)
        total_scenarios = len(all_attackers_summary)
        total_attacker_messages = 0
        
        for summary in all_attackers_summary:
            attacker_type_stats[summary['attacker_type']] += 1
            total_attacker_messages += summary['total_attacker_messages']
            
            print(f"\n시나리오: {summary['scenario']}")
            print(f"  공격자 유형: AttackerType{summary['attacker_type']}")
            print(f"  공격자 sender들: {summary['attacker_senders']}")
            print(f"  공격자 메시지 수: {summary['total_attacker_messages']}개")
            print(f"  JSONlog 파일 수: {summary['jsonlog_files_count']}개")
        
        print(f"\n" + "="*80)
        print("전체 통계")
        print("="*80)
        print(f"처리된 시나리오 수: {total_scenarios}개")
        print(f"총 공격자 메시지 수: {total_attacker_messages}개")
        print("\n공격자 유형별 시나리오 분포:")
        for att_type, count in sorted(attacker_type_stats.items()):
            print(f"  AttackerType{att_type}: {count}개 시나리오")
        
        # 전체 요약 정보를 JSON 파일로 저장
        summary_data = {
            "total_scenarios": total_scenarios,
            "total_attacker_messages": total_attacker_messages,
            "attacker_type_distribution": dict(attacker_type_stats),
            "scenarios": all_attackers_summary
        }
        
        summary_file = os.path.join(output_dir, "all_attackers_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n전체 요약 정보가 {summary_file}에 저장되었습니다.")
        print(f"각 시나리오별 공격자 데이터가 {output_dir} 디렉토리에 저장되었습니다.")
        
        # 공격자 sender들만 따로 정리한 파일 생성
        all_senders_by_type = defaultdict(set)
        for summary in all_attackers_summary:
            att_type = summary['attacker_type']
            for sender in summary['attacker_senders']:
                all_senders_by_type[att_type].add(sender)
        
        senders_summary = {}
        for att_type, senders in all_senders_by_type.items():
            senders_summary[f"AttackerType{att_type}"] = sorted(list(senders))
        
        senders_file = os.path.join(output_dir, "attacker_senders_by_type.json")
        with open(senders_file, 'w', encoding='utf-8') as f:
            json.dump(senders_summary, f, indent=2, ensure_ascii=False)
        
        print(f"공격자 sender 목록이 {senders_file}에 저장되었습니다.")
        print("\n공격자 유형별 sender 목록:")
        for att_type, senders in sorted(senders_summary.items()):
            print(f"  {att_type}: {senders}")
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    extract_attackers_by_directory()
